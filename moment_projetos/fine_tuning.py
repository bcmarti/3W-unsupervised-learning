"""
MOMENT Fine-tuning + Anomaly Detection on the 3W Dataset
=========================================================
Pipeline:
  1. Load 3W parquet files — train ONLY on label=0 (normal) windows
  2. Fine-tune MOMENT's reconstruction backbone + head on normal data
  3. Test on held-out windows (normal + failure) using reconstruction MSE
  4. Evaluate with ROC-AUC and plot anomaly scores

Requirements:
    pip install momentfm pandas pyarrow scikit-learn matplotlib torch

3W dataset directory structure expected:
    3W/
    └── dataset/
        ├── 0/          ← normal instances  (label 0)
        ├── 1/          ← event type 1
        ├── 2/          ← event type 2
        ...
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report
from momentfm import MOMENTPipeline

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — adjust these paths and hyperparameters to your setup
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT   = "./3W/dataset"        # root dir of 3W parquet files
CHECKPOINT_PATH = "moment_3w.pt"       # where to save the fine-tuned model
SEQ_LEN        = 512                   # MOMENT fixed input length (timesteps)
PATCH_LEN      = 8                     # default patch size used by MOMENT
MASK_RATIO     = 0.4                   # fraction of patches masked during training
STRIDE         = 256                   # sliding window stride
BATCH_SIZE     = 32
EPOCHS         = 10
LR             = 1e-4
WEIGHT_DECAY   = 1e-2
THRESHOLD_SIGMA = 2.5                  # threshold = mean + N*std of normal scores
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# 3W sensor columns (8 variables)
SENSORS = [
    "P-PDG", "P-TPT", "T-TPT", "P-MON-CKG",
    "T-JUS-CKG", "P-JUS-CKGL", "T-JUS-CKGL", "QGL",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_instance(path: str) -> pd.DataFrame:
    """Load a single 3W parquet instance and return a clean DataFrame."""
    df = pd.read_parquet(path, engine="pyarrow")
    # The index is the timestamp; reset so it's a regular column
    df = df.reset_index(drop=True)
    # Keep only the sensor columns that exist in this file
    available = [c for c in SENSORS if c in df.columns]
    label_col  = "class" if "class" in df.columns else df.columns[-1]
    df = df[available + [label_col]].rename(columns={label_col: "label"})
    # Interpolate NaNs (very common in 3W) then fill remaining with 0
    df[available] = (
        df[available]
        .interpolate(method="linear", limit_direction="both")
        .fillna(0)
    )
    return df, available


def extract_windows(df: pd.DataFrame, sensors: list, seq_len: int, stride: int):
    """
    Slide a window of length `seq_len` over the instance.
    Returns:
        windows : np.ndarray  shape (N, seq_len, C)
        labels  : np.ndarray  shape (N,)  — 0=normal, 1=anomaly
    """
    arr    = df[sensors].values.astype(np.float32)   # (T, C)
    labels = df["label"].values

    windows, window_labels = [], []
    for start in range(0, len(arr) - seq_len + 1, stride):
        end    = start + seq_len
        window = arr[start:end]
        wlabel = labels[start:end]
        # A window is anomalous if ANY timestep carries a non-zero label
        is_anomaly = int(np.any(wlabel > 0))
        windows.append(window)
        window_labels.append(is_anomaly)

    return np.array(windows, dtype=np.float32), np.array(window_labels)


def collect_all_windows(dataset_root: str, seq_len: int, stride: int,
                        normal_only: bool = False, max_files_per_class: int = None):
    """
    Traverse all event-type subdirectories under dataset_root.
    If normal_only=True, only loads the '0/' directory.
    """
    all_windows, all_labels = [], []
    event_dirs = sorted(glob.glob(os.path.join(dataset_root, "*")))

    for event_dir in event_dirs:
        if not os.path.isdir(event_dir):
            continue
        event_id = os.path.basename(event_dir)
        if normal_only and event_id != "0":
            continue

        parquet_files = sorted(glob.glob(os.path.join(event_dir, "*.parquet")))
        if max_files_per_class:
            parquet_files = parquet_files[:max_files_per_class]

        for path in parquet_files:
            try:
                df, sensors = load_instance(path)
                if len(df) < seq_len:
                    continue
                windows, labels = extract_windows(df, sensors, seq_len, stride)
                all_windows.append(windows)
                all_labels.append(labels)
            except Exception as e:
                print(f"  [WARN] Could not load {path}: {e}")

    if not all_windows:
        raise RuntimeError(f"No windows extracted from {dataset_root}. "
                           "Check your DATASET_ROOT path.")

    return (
        np.concatenate(all_windows, axis=0),
        np.concatenate(all_labels,  axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """
    Wraps pre-extracted windows for MOMENT.
    Each sample:
        x          — (1, seq_len)   single-channel normalized window
        input_mask — (seq_len,)     all-ones (no padding)
        mask       — (seq_len,)     0 = masked patch, 1 = observed
    MOMENT processes one channel at a time, so we expand the C channels
    into separate samples inside __getitem__.
    """

    def __init__(self, windows: np.ndarray, mask_ratio: float = MASK_RATIO):
        """
        windows : (N, seq_len, C)
        After __init__, self.samples contains (N*C) single-channel windows.
        """
        self.mask_ratio = mask_ratio
        self.seq_len    = windows.shape[1]
        self.n_patches  = self.seq_len // PATCH_LEN

        # Flatten to per-channel samples: list of 1-D arrays
        self.samples = []
        for w in windows:                   # w : (seq_len, C)
            for ch in range(w.shape[1]):
                self.samples.append(w[:, ch])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx].copy()        # (seq_len,)

        # Reversible Instance Normalization (per-sample)
        mean = x.mean()
        std  = x.std() + 1e-8
        x    = (x - mean) / std

        # Random patch mask: 1 = observed, 0 = masked
        mask = torch.ones(self.seq_len)
        n_masked = max(1, int(self.mask_ratio * self.n_patches))
        masked_idx = np.random.choice(self.n_patches, n_masked, replace=False)
        for p in masked_idx:
            mask[p * PATCH_LEN:(p + 1) * PATCH_LEN] = 0.0

        return {
            "x":          torch.tensor(x, dtype=torch.float32).unsqueeze(0),  # (1, T)
            "input_mask": torch.ones(self.seq_len),
            "mask":       mask,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────

def build_model():
    print("Loading MOMENT-1-large in reconstruction mode...")
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction"},
    )
    model.init()

    # model.init() leaves the backbone frozen by default, which breaks
    # gradient checkpointing (produces 'Gradients will be None' warning).
    # Explicitly unfreeze everything so gradients flow during fine-tuning.
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")
    return model.to(DEVICE)


def finetune(model, train_windows: np.ndarray):
    """Fine-tune MOMENT on normal-only windows using masked reconstruction loss."""
    dataset    = WindowDataset(train_windows, mask_ratio=MASK_RATIO)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, pin_memory=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(dataloader)
    )

    print(f"\nFine-tuning on {len(dataset):,} single-channel windows "
          f"for {EPOCHS} epochs on {DEVICE}...\n")

    # Suppress use_reentrant warning from gradient checkpointing in MOMENT
    import functools, torch.utils.checkpoint as _ckpt
    _ckpt.checkpoint = functools.partial(_ckpt.checkpoint, use_reentrant=False)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            x          = batch["x"].to(DEVICE)           # (B, 1, T)
            input_mask = batch["input_mask"].to(DEVICE)   # (B, T)
            mask       = batch["mask"].to(DEVICE)         # (B, T)

            output = model(x_enc=x, input_mask=input_mask, mask=mask)

            # TimeseriesOutputs has no .loss — compute MSE on masked positions
            recon      = output.reconstruction          # (B, 1, T)
            masked_pos = (mask == 0).unsqueeze(1)       # (B, 1, T) bool
            loss = torch.nn.functional.mse_loss(
                recon[masked_pos], x[masked_pos]
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch:02d}/{EPOCHS}  loss={avg:.6f}")

    print(f"\nSaving checkpoint to {CHECKPOINT_PATH}")
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANOMALY SCORING (INFERENCE)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_windows(model, windows: np.ndarray, mask_ratio: float = MASK_RATIO) -> np.ndarray:
    """
    For each multi-channel window:
      1. For each channel, mask `mask_ratio` of patches and reconstruct.
      2. Compute MSE between original and reconstruction on masked positions only.
      3. Aggregate across channels with max (most sensitive to any single failing sensor).
    Returns: anomaly_scores shape (N,)
    """
    model.eval()
    scores = []

    for window in windows:                              # window: (seq_len, C)
        channel_mses = []
        seq_len, n_channels = window.shape
        n_patches = seq_len // PATCH_LEN

        for ch in range(n_channels):
            x    = window[:, ch].copy().astype(np.float32)
            mean = x.mean(); std = x.std() + 1e-8
            x    = (x - mean) / std

            # Build a consistent mask for inference
            mask = torch.ones(1, seq_len)
            n_masked = max(1, int(mask_ratio * n_patches))
            masked_patches = np.random.choice(n_patches, n_masked, replace=False)
            for p in masked_patches:
                mask[0, p * PATCH_LEN:(p + 1) * PATCH_LEN] = 0.0

            x_t          = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,T)
            input_mask_t = torch.ones(1, seq_len).to(DEVICE)
            mask_t       = mask.to(DEVICE)

            output = model(x_enc=x_t, input_mask=input_mask_t, mask=mask_t)
            recon  = output.reconstruction.squeeze().cpu().numpy()   # (T,)
            orig   = x

            # MSE only on masked timesteps
            masked_positions = (mask.squeeze().numpy() == 0)
            if masked_positions.any():
                mse = np.mean((orig[masked_positions] - recon[masked_positions]) ** 2)
            else:
                mse = np.mean((orig - recon) ** 2)

            channel_mses.append(mse)

        scores.append(np.max(channel_mses))   # worst-case channel drives the score

    return np.array(scores)


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION + PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(scores, labels, threshold):
    preds = (scores > threshold).astype(int)
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Threshold : {threshold:.6f}")
    print(f"Windows   : {len(labels)} total  |  "
          f"{(labels==0).sum()} normal  |  {(labels==1).sum()} anomaly\n")
    print(classification_report(labels, preds, target_names=["Normal", "Anomaly"]))
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, scores)
        print(f"ROC-AUC : {auc:.4f}")
    else:
        print("[INFO] Only one class present — ROC-AUC not defined.")


def plot_results(scores, labels, threshold, save_path="anomaly_scores.png"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Top: anomaly score over windows
    ax = axes[0]
    normal_idx  = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    ax.scatter(normal_idx,  scores[normal_idx],  s=10, color="steelblue",
               alpha=0.5, label="Normal")
    ax.scatter(anomaly_idx, scores[anomaly_idx], s=10, color="crimson",
               alpha=0.7, label="Anomaly")
    ax.axhline(threshold, color="orange", linewidth=1.5,
               linestyle="--", label=f"Threshold ({threshold:.4f})")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Anomaly Scores per Window")
    ax.legend(loc="upper right")

    # Bottom: ground truth labels
    ax2 = axes[1]
    ax2.fill_between(range(len(labels)), labels, step="mid",
                     color="crimson", alpha=0.4, label="True anomaly")
    preds = (scores > threshold).astype(int)
    ax2.fill_between(range(len(preds)), preds, step="mid",
                     color="orange", alpha=0.3, label="Predicted anomaly")
    ax2.set_ylim(-0.1, 1.4)
    ax2.set_xlabel("Window index")
    ax2.set_ylabel("Label")
    ax2.set_title("True vs Predicted Anomaly Windows")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MOMENT × 3W  —  Fine-tune + Anomaly Detection")
    print("=" * 60)

    # ── Step 1: Load 3W data ──────────────────────────────────────────────
    print("\n[1/4] Loading 3W dataset...")

    # Training: normal windows only
    print("  Loading normal instances (class=0) for training...")
    normal_windows, normal_labels = collect_all_windows(
        DATASET_ROOT, SEQ_LEN, STRIDE, normal_only=True
    )
    print(f"  Normal windows: {len(normal_windows):,}")

    # Testing: sample from ALL event types (including failures)
    print("  Loading all event types for testing (capped per class)...")
    test_windows, test_labels = collect_all_windows(
        DATASET_ROOT, SEQ_LEN, STRIDE,
        normal_only=False, max_files_per_class=5
    )
    print(f"  Test windows : {len(test_windows):,}  "
          f"(normal: {(test_labels==0).sum()}, "
          f"anomaly: {(test_labels==1).sum()})")

    # Train/val split of normal data for threshold calibration
    n_train = int(0.85 * len(normal_windows))
    idx     = np.random.permutation(len(normal_windows))
    train_windows = normal_windows[idx[:n_train]]
    val_windows   = normal_windows[idx[n_train:]]

    # ── Step 2: Build model ───────────────────────────────────────────────
    print("\n[2/4] Building model...")
    model = build_model()

    # ── Step 3: Fine-tune ────────────────────────────────────────────────
    print("\n[3/4] Fine-tuning MOMENT on normal 3W data...")
    if os.path.exists(CHECKPOINT_PATH):
        ans = input(f"  Checkpoint '{CHECKPOINT_PATH}' exists. "
                    "Load it instead of re-training? [y/N]: ").strip().lower()
        if ans == "y":
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
            print("  Checkpoint loaded.")
        else:
            model = finetune(model, train_windows)
    else:
        model = finetune(model, train_windows)

    # ── Step 4: Score & evaluate ─────────────────────────────────────────
    print("\n[4/4] Scoring windows and evaluating...")

    # Calibrate threshold on validation normal windows
    print("  Scoring validation (normal) windows for threshold calibration...")
    val_scores = score_windows(model, val_windows)
    threshold  = val_scores.mean() + THRESHOLD_SIGMA * val_scores.std()
    print(f"  Calibrated threshold (μ + {THRESHOLD_SIGMA}σ): {threshold:.6f}")

    # Score test set
    print("  Scoring test windows...")
    test_scores = score_windows(model, test_windows)

    evaluate(test_scores, test_labels, threshold)
    plot_results(test_scores, test_labels, threshold)


if __name__ == "__main__":
    main()