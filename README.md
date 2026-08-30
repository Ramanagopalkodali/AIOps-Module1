# AIOps Module 1 — Experiment Management & Reproducibility

This repository contains the Module 1 assignment for the AI Operations (AIOps) course, covering
technical debt diagnosis, MLflow experiment tracking, DVC data versioning, and an end-to-end
reproducibility drill.

## Repository Structure

```
AIOps-Module1/
├── environment.yml           # Conda/mamba environment definition
├── mlflow.db                 # MLflow tracking backend (SQLite)
├── mlruns/                   # MLflow run artifacts (models, params, metrics)
├── Q1_Technical_Debt/
│   └── answer.md             # Conceptual analysis of hidden technical debt
├── Q2_MLflow/
│   ├── train_mnist_mlp.py    # Training script (MLP on MNIST) with MLflow logging
│   ├── analysis.md           # Written comparison of the 6 tracked experiments
│   └── results/
│       └── mlflow_comparision.png
├── Q3_DVC/
│   ├── data.dvc              # DVC pointer file (tracks data/raw images)
│   ├── file_list.csv         # CSV manifest of dataset filenames (Git-tracked)
│   ├── rollback.md           # Terminal proof of v1 rollback
│   └── .gitignore            # Excludes DVC-tracked data from Git
├── Q4_Reproducibility/
│   ├── train.py              # Capstone training script
│   └── reproduction.md       # Partner B's reproduction notes
└── .dvc/
    └── config                # DVC remote configuration
```

## 1. Environment Setup

Clone the repo and create the environment:

```bash
git clone https://github.com/Ramanagopalkodali/AIOps-Module1.git
cd AIOps-Module1
mamba env create -f environment.yml   # or: conda env create -f environment.yml
mamba activate aiops                  # replace 'aiops' with the name in environment.yml
```

Install DVC with SSH support (needed to pull the dataset in Q3):

```bash
pip install dvc[ssh]
```

## 2. Q1 — Technical Debt Diagnosis

No code to run. See [`Q1_Technical_Debt/answer.md`](Q1_Technical_Debt/answer.md) for the
conceptual write-up mapping each scenario to a hidden-technical-debt category and proposed
mitigation.

## 3. Q2 — MLflow Experiment Tracking

Runs an MLP classifier on MNIST while logging parameters and metrics to MLflow.

```bash
cd Q2_MLflow
python train_mnist_mlp.py --lr 0.001 --batch-size 32 --epochs 10
```

Repeat with different `--lr` / `--batch-size` values to reproduce the 6 tracked runs.

View and compare runs in the MLflow UI:

```bash
cd ..                       # repo root, so mlflow.db and mlruns/ are found
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000` in a browser. The run-comparison table and written analysis
are in [`Q2_MLflow/analysis.md`](Q2_MLflow/analysis.md).

## 4. Q3 — DVC Data Versioning & Rollback

### Pipeline overview

- **Git** tracks code and the small `data.dvc` pointer file (contains a hash of the dataset).
- **DVC** tracks the actual dataset (images) and stores the real content in a remote — in this
  project, an **SSH remote**.
- `file_list.csv` (a manifest of filenames in the dataset) is tracked directly in Git for easy
  browsing, while the underlying image files are DVC-tracked.

### Pull the dataset

After cloning and activating the environment, fetch the DVC-tracked data:

```bash
dvc pull
```

> **Note:** This requires SSH access to the configured remote (see `.dvc/config`). If you're
> not the original author, point the remote to your own storage location and `dvc push` your
> own copy first — see [Reconfiguring the remote](#reconfiguring-the-remote) below.

### Version history

| Tag | Dataset state              | Command to check out           |
|-----|-----------------------------|----------------------------------|
| `v1` | 1800 rows (base dataset)   | `git checkout v1 && dvc checkout` |
| `v2` | 2801 rows (+ new_labels.zip) | `git checkout main && dvc checkout` |

### Reproduce the versioning steps from scratch

```bash
cd Q3_DVC
dvc get https://github.com/iterative/dataset-registry tutorials/versioning/data.zip
mkdir -p data/raw
unzip data.zip -d data/raw && rm -f data.zip

echo "filename" > file_list.csv
find data/raw -type f | sed 's|.*/data/raw/||' >> file_list.csv   # 1801 lines = 1800 rows + header

cd ..
dvc add Q3_DVC/data/raw
git add Q3_DVC/data/raw.dvc Q3_DVC/file_list.csv Q3_DVC/.gitignore
git commit -m "Q3: data v1 (1800 rows + header)"
git tag -a v1 -m "Data v1 - 1800 rows"
dvc push
```

### Rollback demonstration

```bash
git checkout v1
dvc checkout
wc -l Q3_DVC/file_list.csv   # prints 1801, matching v1 exactly

git checkout main
dvc checkout                 # restores v2 (2801 rows)
```

Full terminal output proving the rollback is documented in
[`Q3_DVC/rollback.md`](Q3_DVC/rollback.md).

### Reconfiguring the remote

The current remote is an SSH server (see `.dvc/config`). To point it at your own storage:

```bash
dvc remote remove sshremote
dvc remote add --default sshremote ssh://<user>@<host>/path/to/storage
dvc push
```

## 5. Q4 — End-to-End Reproducibility Drill

This is a **paired exercise**. Partner A trains and versions the experiment; Partner B clones
and reproduces it independently.

**Partner A:**
```bash
cd Q4_Reproducibility
python train.py --seed 42
# Registers the model in MLflow, tags it with the git commit hash, and transitions it to "Staging"
git add Q4_Reproducibility/ Q4_Reproducibility/*.dvc
git commit -m "Q4: train and version experiment"
dvc push
```

**Partner B (using only the commands below, no other communication):**
```bash
git clone <repo-url>
cd AIOps-Module1
git checkout <commit-hash>
dvc checkout
mamba env create -f environment.yml
mamba activate aiops
python Q4_Reproducibility/train.py --seed 42
```

Partner B then logs a note in the MLflow run documenting whether metrics matched within
tolerance — see [`Q4_Reproducibility/reproduction.md`](Q4_Reproducibility/reproduction.md) for
the outcome.

## Notes

- The DVC SSH remote in this project runs on the assignment author's own machine (accessed via
  `ssh://` protocol), satisfying the assignment's requirement of a genuine SSH remote as opposed
  to a `local` DVC remote type.
- MLflow tracking data (`mlflow.db`, `mlruns/`) is committed directly to Git for grading
  convenience; in a production setting this would typically also be excluded and reconstructed
  via a tracking server.
