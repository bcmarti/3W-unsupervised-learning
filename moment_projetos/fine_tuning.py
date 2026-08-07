"""
MOMENT Fine-tuning + Anomaly Detection
=======================================
Pipeline:
  1. Train on UNLABELED data (flat folder of parquet/csv files — no labels needed)
  2. Validate reconstruction loss on 3W normal-only windows during training
  3. Test + evaluate on 3W labeled data (normal vs failure) using reconstruction MSE
  4. Calibrate threshold on 3W normal windows, then score all test windows

Directory structure expected:

  UNLABELED_ROOT/          ← training data (no subfolders needed)
      well_001.parquet
      well_002.parquet
      ...

  W3_ROOT/                 ← 3W dataset (subfolders by event type)
      0/                   ← normal instances
          WELL-00001_20140106224024.parquet
          ...
      1/                   ← event type 1 (failure)
      2/                   ← event type 2
      ...
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report
from momentfm import MOMENTPipeline

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust paths and hyperparameters to your setup
# ─────────────────────────────────────────────────────────────────────────────

UNLABELED_ROOT  = "./unlabeled"       # flat folder with training parquet/csv files
W3_ROOT         = "./3W/dataset"      # 3W dataset root (subfolders 0/, 1/, 2/, ...)
CHECKPOINT_PATH = "moment_finetuned.pt"

# SEQ_LEN must be a multiple of PATCH_LEN (8).
# At 1-min resolution (RESAMPLE_3W=True), 512 timesteps = ~8.5 hours per window,
# which is longer than most 3W instances after resampling (typically 1-4 hours).
# 128 = ~2 hours per window — a better fit for minute-resolution 3W data.
# At second resolution (RESAMPLE_3W=False), 512 = ~8.5 minutes, keep it.
SEQ_LEN          = 128    # MOMENT fixed input length (timesteps)
PATCH_LEN        = 8      # MOMENT default patch size
MASK_RATIO       = 0.4    # fraction of patches masked during training
STRIDE           = 64     # sliding window stride

BATCH_SIZE       = 64  if torch.cuda.is_available() else 32
EPOCHS           = 20  if torch.cuda.is_available() else 5
LR               = 1e-4
WEIGHT_DECAY     = 1e-2
THRESHOLD_SIGMA  = 2.5    # anomaly threshold = mean + N * std of normal val scores

# Cap files per 3W class for test set to avoid class imbalance (None = use all)
MAX_3W_TEST_FILES_PER_CLASS = None

# Resample 3W data to 1-minute intervals to match unlabeled data frequency.
# Sensors are averaged within each minute; labels use max (any failure second
# in a minute marks the whole minute as anomalous).
# Set to False if your unlabeled data is also at second-level resolution.
RESAMPLE_3W = True

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sensor columns shared between unlabeled data and 3W
# Edit this list to match your unlabeled dataset's column names
SENSORS = [
    "P-PDG", "P-TPT", "T-TPT", "P-MON-CKG",
    "T-JUS-CKG", "P-JUS-CKGL", "T-JUS-CKGL", "QGL",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _read_file(path: str) -> pd.DataFrame:
    """Read a parquet or csv file into a DataFrame."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    elif ext in (".csv", ".tsv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def load_unlabeled_instance(path: str):
    """
    Load a single unlabeled file.
    Returns (DataFrame with sensor columns, list of available sensor names).
    No label column expected or required.
    """
    df = _read_file(path).reset_index(drop=True)
    available = [c for c in SENSORS if c in df.columns]
    if not available:
        raise ValueError(f"None of the expected SENSORS found in {path}. "
                         f"Columns present: {list(df.columns)}")
    df = df[available].copy()
    # Replace inf/-inf with NaN before interpolating so they don't
    # propagate and cause overflow when casting to float32.
    df[available] = df[available].replace([np.inf, -np.inf], np.nan)
    df[available] = (
        df[available]
        .interpolate(method="linear", limit_direction="both")
        .fillna(0)
    )
    # Hard clip to float32 safe range to prevent overflow in normalization
    df[available] = df[available].clip(-3.4e38, 3.4e38)
    return df, available


def load_3w_instance(path: str, resample: bool = None):
    """
    Load a single 3W parquet instance.
    If resample=True (or RESAMPLE_3W=True), downsample from seconds to 1-minute
    intervals: sensors are averaged, label uses max so any failure second within
    a minute marks the whole minute as anomalous.
    Returns (DataFrame with sensor + label columns, list of available sensor names).
    """
    if resample is None:
        resample = RESAMPLE_3W

    # 3W stores timestamps as the DataFrame index — keep it for resampling
    df = _read_file(path)
    available = [c for c in SENSORS if c in df.columns]
    label_col = "class" if "class" in df.columns else df.columns[-1]
    df = df[available + [label_col]].rename(columns={label_col: "label"})

    if resample:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # Sensors: mean per minute; label: max per minute (conservative —
        # one failure second makes the whole minute anomalous)
        sensor_resampled = df[available].resample("1min").mean()
        label_resampled  = df[["label"]].resample("1min").max()
        df = pd.concat([sensor_resampled, label_resampled], axis=1)

    df = df.reset_index(drop=True)
    df[available] = df[available].replace([np.inf, -np.inf], np.nan)
    df[available] = (
        df[available]
        .interpolate(method="linear", limit_direction="both")
        .fillna(0)
    )
    df[available] = df[available].clip(-3.4e38, 3.4e38)
    df["label"] = df["label"].fillna(0).astype(int)
    return df, available


def extract_windows_unlabeled(df: pd.DataFrame, sensors: list,
                               seq_len: int, stride: int):
    """Extract sliding windows from an unlabeled file. Returns (N, seq_len, C)."""
    arr     = df[sensors].values.astype(np.float32)
    windows = []
    for start in range(0, len(arr) - seq_len + 1, stride):
        windows.append(arr[start:start + seq_len])
    return np.array(windows, dtype=np.float32) if windows else np.empty((0, seq_len, len(sensors)))


def extract_windows_3w(df: pd.DataFrame, sensors: list, seq_len: int, stride: int):
    """
    Extract sliding windows from a labeled 3W instance.
    Returns:
        windows : (N, seq_len, C)
        labels  : (N,)  — 0 = normal, 1 = anomaly
    """
    arr    = df[sensors].values.astype(np.float32)
    labels = df["label"].values
    windows, window_labels = [], []
    for start in range(0, len(arr) - seq_len + 1, stride):
        end    = start + seq_len
        window = arr[start:end]
        wlabel = labels[start:end]
        windows.append(window)
        window_labels.append(int(np.any(wlabel > 0)))
    if not windows:
        return np.empty((0, seq_len, len(sensors))), np.empty(0, dtype=int)
    return np.array(windows, dtype=np.float32), np.array(window_labels)


def collect_unlabeled_windows(root: str, seq_len: int, stride: int):
    """
    Load all parquet/csv files from a flat folder (no subfolders expected).
    Returns: windows (N, seq_len, C)
    """
    patterns = [
        os.path.join(root, "*.parquet"),
        os.path.join(root, "*.csv"),
        os.path.join(root, "*.tsv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(files)

    if not files:
        raise RuntimeError(
            f"No parquet/csv files found in '{root}'.\n"
            f"Check your UNLABELED_ROOT path."
        )

    all_windows = []
    for path in files:
        try:
            df, sensors = load_unlabeled_instance(path)
            if len(df) < seq_len:
                continue
            windows = extract_windows_unlabeled(df, sensors, seq_len, stride)
            if len(windows):
                all_windows.append(windows)
        except Exception as e:
            print(f"  [WARN] Skipping {path}: {e}")

    if not all_windows:
        raise RuntimeError(f"No windows could be extracted from '{root}'.")

    result = np.concatenate(all_windows, axis=0)
    print(f"  Unlabeled training windows: {len(result):,}  "
          f"(from {len(files)} files)")
    return result


def collect_3w_windows(root: str, seq_len: int, stride: int,
                        normal_only: bool = False,
                        max_files_per_class: int = None):
    """
    Traverse 3W subdirectories (0/, 1/, 2/, ...).
    Returns: windows (N, seq_len, C), labels (N,)
    """
    all_windows, all_labels = [], []
    event_dirs = sorted(glob.glob(os.path.join(root, "*")))

    for event_dir in event_dirs:
        if not os.path.isdir(event_dir):
            continue
        event_id = os.path.basename(event_dir)
        if normal_only and event_id != "0":
            continue

        files = sorted(glob.glob(os.path.join(event_dir, "*.parquet")))
        if max_files_per_class:
            files = files[:max_files_per_class]

        for path in files:
            try:
                df, sensors = load_3w_instance(path)
                if len(df) < seq_len:
                    continue
                windows, labels = extract_windows_3w(df, sensors, seq_len, stride)
                if len(windows):
                    all_windows.append(windows)
                    all_labels.append(labels)
            except Exception as e:
                print(f"  [WARN] Skipping {path}: {e}")

    if not all_windows:
        raise RuntimeError(f"No 3W windows extracted from '{root}'. "
                           "Check your W3_ROOT path.")

    return (
        np.concatenate(all_windows, axis=0),
        np.concatenate(all_labels,  axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """
    Accepts windows of shape (N, seq_len, C).
    Expands each multi-channel window into C independent single-channel samples,
    since MOMENT processes one channel at a time.
    Each sample returns:
        x          — (1, seq_len)  normalized signal
        input_mask — (seq_len,)    all-ones (no padding)
        mask       — (seq_len,)    0 = masked patch, 1 = observed
    """

    def __init__(self, windows: np.ndarray, mask_ratio: float = MASK_RATIO):
        self.mask_ratio = mask_ratio
        self.seq_len    = windows.shape[1]
        self.n_patches  = self.seq_len // PATCH_LEN

        self.samples = []
        for w in windows:                   # w: (seq_len, C)
            for ch in range(w.shape[1]):
                self.samples.append(w[:, ch])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x    = self.samples[idx].copy()
        # Work in float64 for mean/std — float32 overflows with large sensor values
        x    = x.astype(np.float64)
        x    = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mean = x.mean()
        std  = x.std() + 1e-8
        x    = (x - mean) / std
        # Clamp normalized output to [-10, 10] — values beyond this are
        # outliers that will still cause inf in float32 after conversion
        x    = np.clip(x, -10.0, 10.0)

        mask      = torch.ones(self.seq_len)
        n_masked  = max(1, int(self.mask_ratio * self.n_patches))
        masked_idx = np.random.choice(self.n_patches, n_masked, replace=False)
        for p in masked_idx:
            mask[p * PATCH_LEN:(p + 1) * PATCH_LEN] = 0.0

        return {
            "x":          torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            "input_mask": torch.ones(self.seq_len),
            "mask":       mask,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model():
    print("Loading MOMENT-1-large in reconstruction mode...")
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction"},
    )
    model.init()

    # Unfreeze all parameters — model.init() freezes backbone by default,
    # which breaks gradient checkpointing and prevents learning.
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} parameters  |  device: {DEVICE}")
    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(model, batch):
    """Shared loss computation used in both train and val steps."""
    x          = batch["x"].to(DEVICE)
    input_mask = batch["input_mask"].to(DEVICE)
    mask       = batch["mask"].to(DEVICE)

    output     = model(x_enc=x, input_mask=input_mask, mask=mask)
    recon      = output.reconstruction          # (B, 1, T)
    masked_pos = (mask == 0).unsqueeze(1)       # (B, 1, T) bool
    return F.mse_loss(recon[masked_pos], x[masked_pos])


def finetune(model, train_windows: np.ndarray, val_windows: np.ndarray):
    """
    Fine-tune MOMENT using masked reconstruction on unlabeled training windows.
    Validation loss is computed on 3W normal windows each epoch to monitor
    how well the model generalises to the target domain.
    """
    train_dataset = WindowDataset(train_windows, mask_ratio=MASK_RATIO)
    val_dataset   = WindowDataset(val_windows,   mask_ratio=MASK_RATIO)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader)
    )

    print(f"\nTraining:   {len(train_dataset):,} single-channel windows "
          f"({len(train_windows):,} multi-ch windows × {train_windows.shape[2]} channels)")
    print(f"Validation: {len(val_dataset):,} single-channel windows "
          f"(3W normal, used for val loss + threshold calibration)")
    print(f"Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Device: {DEVICE}\n")

    # Silence use_reentrant warning from MOMENT's gradient checkpointing
    import functools, torch.utils.checkpoint as _ckpt
    _ckpt.checkpoint = functools.partial(_ckpt.checkpoint, use_reentrant=False)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate on 3W normal windows ────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += compute_loss(model, batch).item()
        val_loss /= len(val_loader)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            tag = "  ← best"
        else:
            tag = ""

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}{tag}")

    print(f"\nBest val loss: {best_val_loss:.6f} — checkpoint: {CHECKPOINT_PATH}")
    # Reload best weights (only if checkpoint was actually saved)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE,
                                         weights_only=True))
    else:
        print("  [WARN] No checkpoint saved (loss was NaN every epoch). "
              "Check your data for inf/nan values.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANOMALY SCORING (INFERENCE)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_windows(model, windows: np.ndarray,
                  mask_ratio: float = MASK_RATIO) -> np.ndarray:
    """
    Score each window with reconstruction MSE on masked patches.
    Aggregates across channels with max() — any single failing sensor
    raises the anomaly score.
    Returns: anomaly_scores shape (N,)
    """
    model.eval()
    scores    = []
    n_patches = SEQ_LEN // PATCH_LEN

    for window in windows:                          # (seq_len, C)
        channel_mses = []
        for ch in range(window.shape[1]):
            x    = window[:, ch].copy().astype(np.float64)
            x    = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            mean = x.mean(); std = x.std() + 1e-8
            x    = (x - mean) / std
            x    = np.clip(x, -10.0, 10.0)

            mask      = torch.ones(1, SEQ_LEN)
            n_masked  = max(1, int(mask_ratio * n_patches))
            for p in np.random.choice(n_patches, n_masked, replace=False):
                mask[0, p * PATCH_LEN:(p + 1) * PATCH_LEN] = 0.0

            x_t    = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(DEVICE)
            imask  = torch.ones(1, SEQ_LEN).to(DEVICE)
            mask_t = mask.to(DEVICE)

            output = model(x_enc=x_t, input_mask=imask, mask=mask_t)
            recon  = output.reconstruction.squeeze().cpu().numpy()

            masked = (mask.squeeze().numpy() == 0)
            mse    = np.mean((x[masked] - recon[masked]) ** 2) if masked.any() \
                     else np.mean((x - recon) ** 2)
            channel_mses.append(mse)

        scores.append(np.max(channel_mses))

    return np.array(scores)


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION + PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(scores, labels, threshold):
    preds = (scores > threshold).astype(int)
    print("\n" + "=" * 52)
    print("EVALUATION RESULTS  (3W test set)")
    print("=" * 52)
    print(f"Threshold : {threshold:.6f}  (μ + {THRESHOLD_SIGMA}σ of val normal scores)")
    print(f"Windows   : {len(labels):,} total  |  "
          f"{(labels==0).sum():,} normal  |  {(labels==1).sum():,} anomaly\n")
    print(classification_report(labels, preds, target_names=["Normal", "Anomaly"]))
    if len(np.unique(labels)) > 1:
        if np.isnan(scores).any():
            print("[WARN] ROC-AUC skipped — scores contain NaN. "
                  "Check data for remaining inf/nan values.")
        else:
            auc = roc_auc_score(labels, scores)
            print(f"ROC-AUC : {auc:.4f}")
    else:
        print("[INFO] Only one class in test set — ROC-AUC not defined.")


def plot_results(scores, labels, threshold, save_path="anomaly_scores.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    normal_idx  = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    ax1.scatter(normal_idx,  scores[normal_idx],  s=8, color="steelblue",
                alpha=0.5, label="Normal")
    ax1.scatter(anomaly_idx, scores[anomaly_idx], s=8, color="crimson",
                alpha=0.7, label="Anomaly (3W)")
    ax1.axhline(threshold, color="orange", linewidth=1.5, linestyle="--",
                label=f"Threshold ({threshold:.4f})")
    ax1.set_ylabel("Reconstruction MSE")
    ax1.set_title("Anomaly Scores per Window  (3W test set)")
    ax1.legend(loc="upper right")

    preds = (scores > threshold).astype(int)
    ax2.fill_between(range(len(labels)), labels, step="mid",
                     color="crimson", alpha=0.4, label="True anomaly")
    ax2.fill_between(range(len(preds)),  preds,  step="mid",
                     color="orange",  alpha=0.3, label="Predicted anomaly")
    ax2.set_ylim(-0.1, 1.4)
    ax2.set_xlabel("Window index")
    ax2.set_ylabel("Label")
    ax2.set_title("True vs Predicted Anomaly Windows")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MOMENT  —  Unlabeled pre-training + 3W anomaly detection")
    print("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("\n[1/4] Loading datasets...")

    # Training: unlabeled data (flat folder, no label column required)
    print(f"  Unlabeled training data  ← {UNLABELED_ROOT}")
    train_windows = collect_unlabeled_windows(UNLABELED_ROOT, SEQ_LEN, STRIDE)

    # 3W normal windows: split into val (threshold calibration + val loss)
    # and test (final evaluation alongside failure windows)
    print(f"\n  3W dataset               ← {W3_ROOT}")
    print("  Loading 3W normal instances...")
    normal_windows, _ = collect_3w_windows(
        W3_ROOT, SEQ_LEN, STRIDE, normal_only=True
    )
    print(f"  3W normal windows: {len(normal_windows):,}")

    # 85% val (used during training for loss monitoring + threshold)
    # 15% held out for the final test set alongside failure windows
    idx      = np.random.permutation(len(normal_windows))
    n_val    = int(0.85 * len(normal_windows))
    val_windows        = normal_windows[idx[:n_val]]
    test_normal_windows = normal_windows[idx[n_val:]]
    test_normal_labels  = np.zeros(len(test_normal_windows), dtype=int)

    # Failure windows for the test set
    print("  Loading 3W failure instances...")
    failure_windows, failure_labels = collect_3w_windows(
        W3_ROOT, SEQ_LEN, STRIDE,
        normal_only=False,
        max_files_per_class=MAX_3W_TEST_FILES_PER_CLASS,
    )
    # Keep only the actually-anomalous windows from failure files
    anomaly_mask    = failure_labels == 1
    failure_windows = failure_windows[anomaly_mask]
    failure_labels  = failure_labels[anomaly_mask]
    print(f"  3W failure windows (anomaly label=1): {len(failure_windows):,}")

    # Combine into final test set
    test_windows = np.concatenate([test_normal_windows, failure_windows], axis=0)
    test_labels  = np.concatenate([test_normal_labels,  failure_labels],  axis=0)
    print(f"\n  Final test set: {len(test_windows):,} windows  "
          f"(normal: {(test_labels==0).sum():,}, "
          f"anomaly: {(test_labels==1).sum():,})")

    # ── Step 2: Build model ───────────────────────────────────────────────
    print("\n[2/4] Building model...")
    model = build_model()

    # ── Step 3: Fine-tune ─────────────────────────────────────────────────
    print("\n[3/4] Fine-tuning on unlabeled data / validating on 3W normal...")
    if os.path.exists(CHECKPOINT_PATH):
        ans = input(f"\n  Checkpoint '{CHECKPOINT_PATH}' already exists.\n"
                    "  Load it and skip training? [y/N]: ").strip().lower()
        if ans == "y":
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
            print("  Checkpoint loaded — skipping training.")
        else:
            model = finetune(model, train_windows, val_windows)
    else:
        model = finetune(model, train_windows, val_windows)

    # ── Step 4: Score & evaluate ──────────────────────────────────────────
    print("\n[4/4] Scoring and evaluating on 3W test set...")

    # Threshold from val (3W normal) windows
    print("  Scoring validation windows for threshold calibration...")
    val_scores = score_windows(model, val_windows)
    threshold  = val_scores.mean() + THRESHOLD_SIGMA * val_scores.std()
    print(f"  Threshold (μ + {THRESHOLD_SIGMA}σ): {threshold:.6f}")

    print("  Scoring test windows...")
    test_scores = score_windows(model, test_windows)

    evaluate(test_scores, test_labels, threshold)
    plot_results(test_scores, test_labels, threshold)


if __name__ == "__main__":
    main()
