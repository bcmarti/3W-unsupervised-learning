# 3W-unsupervised-learning

This project sets up a **reproducible Conda environment** for running unsupervised learning experiments using:

* **MOMENT** (time-series foundation model)
* **3W Toolkit** (Petrobras 3W dataset utilities)

The setup is intentionally strict: **Conda manages all dependencies**, while **pip is used only to install project code** (without pulling dependencies).

---

## Prerequisites

Before starting, make sure you have:

* **Conda installed** (Miniconda or Anaconda)
* **Git installed**
* A Linux or macOS environment (tested on Linux)

You can verify Conda with:

```bash
conda --version
```

---

## Repository layout

The `3W` repository **must be cloned one directory above** this project.

### Example directory structure

```
projects/
├── 3W/
│   └── (ThreeWToolkit source code)
│
├── 3W-unsupervised-learning/
│   ├── environment.yml
│   ├── setup.sh
│   └── README.md
```

### Clone repositories

```bash
cd projects

git clone https://github.com/petrobras/3W.git

git clone <THIS_REPOSITORY_URL>
```

---

## Environment setup

The entire setup is automated via `setup.sh`.

### What the script does

1. Creates the Conda environment from `environment.yml`
2. Activates the environment
3. Installs project code (`3W` and `momentfm`) using `pip --no-deps`

### Run setup

From inside `3W-unsupervised-learning`:

```bash
bash setup.sh
```

This will create and configure the Conda environment named:

```
3W-unsupervised
```

---

## Activating the environment

After installation:

```bash
conda activate 3W-unsupervised
```

---

## Verifying the installation

You can quickly test that everything is working:

```bash
python - << 'EOF'
import torch
import momentfm
import ThreeWToolkit

print("Setup successful")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
```

---

## Notes

* **Do not install project dependencies with pip**
* All numerical, CUDA, and ML dependencies are managed by **Conda**
* Pip is used strictly to install source code
* This avoids dependency conflicts (e.g., NumPy, CUDA, PyTorch)

---

## Troubleshooting

If you encounter `ModuleNotFoundError`:

* Add the missing package to `environment.yml`
* Update the environment using:

```bash
conda env update -f environment.yml
```

---
