# ============================================================================
# 3W TOOLKIT — UNLABELED DATA
# ============================================================================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import (
    UnlabeledParquetDatasetConfig,
    UnlabeledParquetDataset,
)

from ThreeWToolkit.preprocessing import (
    CleanSignalsConfig,
    ImputeMissingConfig,
    NormalizeConfig,
    SequentialPreprocessingAdapterConfig,
)

from ThreeWToolkit.dataset import ParquetDatasetConfig, TransformConfig
from ThreeWToolkit.feature_extraction import WindowingConfig

# ============================================================================
# CONFIGURAÇÃO DOS DADOS NÃO ROTULADOS
# ============================================================================

DATA_PATH = "/home/bruno.martins/dataset"

SIGNAL_COLUMNS = None
RECURSIVE = True

IMPUTE_STRATEGY = "mean"
NORMALIZE_NORM = "l2"

WINDOW_SIZE = 128
OVERLAP = 0.5
PAD_LAST_WINDOW = False
PAD_VALUE = 0.0


# ============================================================================
# 1. LEITURA DO DATASET UNLABELED
# ============================================================================

config = UnlabeledParquetDatasetConfig(
    path=DATA_PATH,
    columns=SIGNAL_COLUMNS,
    recursive=RECURSIVE,
)

toolkit_dataset = UnlabeledParquetDataset(
    config
)

print("Dataset criado com sucesso!")

print(
    f"Pasta: {toolkit_dataset.root.resolve()}"
)

print(
    f"Quantidade de arquivos Parquet: "
    f"{len(toolkit_dataset)}"
)


# ============================================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================================

pipeline = SequentialPreprocessingAdapterConfig(
    steps=[
        CleanSignalsConfig(),
        ImputeMissingConfig(
            strategy=IMPUTE_STRATEGY
        ),
        NormalizeConfig(
            norm=NORMALIZE_NORM
        ),
    ]
)

transformer = TransformConfig(
    pre_processing=pipeline
).build()

transformer.fit(
    toolkit_dataset
)

transformed_ds = transformer.transform(
    toolkit_dataset
)

print(
    "Pipeline applied successfully."
)


# ============================================================================
# 3. WINDOWING
# ============================================================================

windowing_transformer = TransformConfig(
    feature_extraction=WindowingConfig(
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        pad_last_window=PAD_LAST_WINDOW,
        pad_value=PAD_VALUE,
    )
).build()

windowing_transformer.fit(
    transformed_ds
)

windowed_dataset = (
    windowing_transformer.transform(
        transformed_ds
    )
)

print(
    "[OK] Windowing executado."
)