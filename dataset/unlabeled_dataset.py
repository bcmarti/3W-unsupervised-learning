from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

@dataclass
class DatasetOutputs:
    """Saída de uma instância do dataset não rotulado."""

    signal: pd.DataFrame
    label: None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class UnlabeledParquetDatasetConfig:
    """Configuração do dataset real não rotulado."""

    path: Path | str
    columns: Optional[Sequence[str]] = None
    recursive: bool = True
    parquet_engine: str = "pyarrow"

    def __post_init__(self):
        self.path = Path(self.path)


class UnlabeledParquetDataset:
    """Dataset para múltiplos Parquets reais e não rotulados."""

    def __init__(self, config: UnlabeledParquetDatasetConfig):
        self.config = config
        self.root = Path(config.path)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Pasta do dataset não encontrada: {self.root.resolve()}"
            )

        if not self.root.is_dir():
            raise NotADirectoryError(
                f"DATA_PATH precisa ser uma pasta: {self.root.resolve()}"
            )

        # Descobre todos os Parquets disponíveis.
        pattern = "**/*.parquet" if config.recursive else "*.parquet"
        self.files_events = sorted(self.root.glob(pattern))

        if not self.files_events:
            raise FileNotFoundError(
                f"Nenhum arquivo .parquet encontrado em: {self.root.resolve()}"
            )

        # Remove duplicatas caso algum caminho seja repetido.
        self.files_events = list(dict.fromkeys(self.files_events))

    def __len__(self) -> int:
        return len(self.files_events)

    def __getitem__(self, idx: int) -> DatasetOutputs:
        return self.load_file(idx)

    def _relative_path(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def load_file(self, idx: int) -> DatasetOutputs:
        """Carrega uma instância Parquet sem tentar obter rótulos."""

        if not isinstance(idx, (int, np.integer)):
            raise TypeError("idx precisa ser um inteiro.")

        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Índice {idx} fora do intervalo [0, {len(self) - 1}]."
            )

        file_path = self.files_events[idx]

        df = pd.read_parquet(
            file_path,
            engine=self.config.parquet_engine,
        )

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"O arquivo {file_path} não foi carregado como DataFrame."
            )

        # Seleção opcional de sensores/variáveis.
        if self.config.columns is not None:
            requested = list(self.config.columns)
            missing = [c for c in requested if c not in df.columns]

            if missing:
                raise ValueError(
                    f"Colunas ausentes em {self._relative_path(file_path)}: "
                    f"{missing}"
                )

            signal_df = df.loc[:, requested].copy()
        else:
            signal_df = df.copy()

        metadata = {
            "file_name": file_path.name,
            "relative_path": self._relative_path(file_path),
            "absolute_path": str(file_path.resolve()),
            "event_type": "real",
            "event_class": None,
            "labeled": False,
            "n_rows": len(signal_df),
            "n_columns": len(signal_df.columns),
            "columns": signal_df.columns.tolist(),
        }

        return DatasetOutputs(
            signal=signal_df,
            label=None,
            metadata=metadata,
        )

    def get_file_list(self) -> list[Path]:
        """Retorna a lista dos arquivos Parquet encontrados."""
        return self.files_events.copy()

    def schema_summary(self, max_files: Optional[int] = None) -> pd.DataFrame:
        """Resume nome, tamanho, número de linhas e colunas dos arquivos."""

        files = self.files_events
        if max_files is not None:
            files = files[:max_files]

        rows = []

        for path in files:
            df = pd.read_parquet(path, engine=self.config.parquet_engine)

            rows.append({
                "file_name": path.name,
                "relative_path": self._relative_path(path),
                "size_MB": path.stat().st_size / (1024 ** 2),
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": ", ".join(map(str, df.columns)),
                "dtypes": ", ".join(
                    f"{c}:{df[c].dtype}" for c in df.columns
                ),
            })

        return pd.DataFrame(rows)

    def validate_files(
        self,
        max_files: Optional[int] = None,
        require_same_columns: bool = False,
        allow_empty: bool = False,
    ) -> pd.DataFrame:
        """Valida se os Parquets podem ser lidos e têm estrutura coerente."""

        files = self.files_events
        if max_files is not None:
            files = files[:max_files]

        results = []

        reference_columns = None

        for path in files:
            result = {
                "file_name": path.name,
                "relative_path": self._relative_path(path),
                "read_ok": False,
                "non_empty": False,
                "n_rows": 0,
                "n_columns": 0,
                "has_nan": False,
                "same_columns": True,
                "error": None,
            }

            try:
                df = pd.read_parquet(
                    path,
                    engine=self.config.parquet_engine,
                )

                result["read_ok"] = True
                result["n_rows"] = len(df)
                result["n_columns"] = len(df.columns)
                result["non_empty"] = len(df) > 0
                result["has_nan"] = bool(df.isna().any().any())

                current_columns = list(df.columns)

                if reference_columns is None:
                    reference_columns = current_columns
                else:
                    result["same_columns"] = (
                        current_columns == reference_columns
                    )

                if not allow_empty and len(df) == 0:
                    result["error"] = "Arquivo vazio."

                if (
                    require_same_columns
                    and not result["same_columns"]
                ):
                    result["error"] = "Colunas diferentes do primeiro arquivo."

            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"

            results.append(result)

        return pd.DataFrame(results)

    def load_instances_by_variable(
        self,
        variables: Optional[list[str]] = None,
    ) -> dict[str, list[np.ndarray]]:
        """Retorna {variável: [array_da_instância, ...]}.

        Arquivos que não possuem uma variável solicitada são ignorados para
        aquela variável.
        """

        if variables is None:
            if self.config.columns is None:
                # Descobre as colunas a partir do primeiro arquivo.
                first = self.load_file(0).signal
                variables = first.columns.tolist()
            else:
                variables = list(self.config.columns)

        data_map: dict[str, list[np.ndarray]] = {
            var: [] for var in variables
        }

        for idx in range(len(self)):
            signal_df = self.load_file(idx).signal

            for var in variables:
                if var in signal_df.columns:
                    values = signal_df[var].to_numpy()
                    data_map[var].append(values)

        return data_map

    def describe(self, max_files: Optional[int] = None) -> pd.DataFrame:
        """Estatísticas básicas das variáveis numéricas."""

        files = self.files_events
        if max_files is not None:
            files = files[:max_files]

        rows = []

        for path in files:
            df = pd.read_parquet(
                path,
                engine=self.config.parquet_engine,
            )

            numeric = df.select_dtypes(include=np.number)

            for column in numeric.columns:
                values = numeric[column].to_numpy(dtype=float)

                rows.append({
                    "file_name": path.name,
                    "variable": column,
                    "n": len(values),
                    "min": np.nanmin(values) if len(values) else np.nan,
                    "max": np.nanmax(values) if len(values) else np.nan,
                    "mean": np.nanmean(values) if len(values) else np.nan,
                    "std": np.nanstd(values) if len(values) else np.nan,
                    "nan_count": int(np.isnan(values).sum()),
                })

        return pd.DataFrame(rows)