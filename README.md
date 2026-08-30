# AIOps Module 1 — Experiment Management & Reproducibility

This repository contains the implementation and evidence for **Module 1 of the AI Operations (AIOps) course**.

The assignment focuses on:

* ML technical debt diagnosis
* MLflow experiment tracking and comparison
* DVC data versioning and rollback
* End-to-end ML reproducibility

---

# Assignment Overview

The assignment consists of four questions:

| Question  | Topic                            |  Marks |
| --------- | -------------------------------- | -----: |
| Q1        | Technical Debt Diagnosis         |     10 |
| Q2        | MLflow Experiment Comparison     |     15 |
| Q3        | DVC Data Versioning & Rollback   |     10 |
| Q4        | End-to-End Reproducibility Drill |     15 |
| **Total** |                                  | **50** |

---

# Repository Structure

```text
AIOps-Module1/
│
├── Q1_Technical_Debt/
│   └── answer.md
│
├── Q2_MLflow/
│   ├── train_mnist_mlp.py
│   ├── analysis.md
│   └── results/
│       └── mlflow_comparision.png
│
├── Q3_DVC/
│   ├── data.dvc
│   ├── file_list.csv
│   ├── rollback.md
│   └── .gitignore
│
├── Q4_Reproducibility/
│   ├── train.py
│   └── reproduction.md
│
├── .dvc/
│   └── config
│
├── .dvcignore
├── .gitignore
├── environment.yml
└── README.md
```

---

# Prerequisites

The project uses:

* Python
* Git
* DVC
* MLflow
* Mamba / Conda

The required environment is defined in:

```text
environment.yml
```

---

# Environment Setup

## 1. Clone the repository

```bash
git clone https://github.com/Ramanagopalkodali/AIOps-Module1.git
cd AIOps-Module1
```

## 2. Create the environment

Using Mamba:

```bash
mamba env create -f environment.yml
mamba activate aiops
```

Alternatively, using Conda:

```bash
conda env create -f environment.yml
conda activate aiops
```

## 3. Install DVC

```bash
pip install "dvc[ssh]"
```

## 4. Verify the installation

```bash
python --version
mlflow --version
dvc --version
```

---

# Question 1 — Technical Debt Diagnosis

## Objective

Q1 identifies the hidden technical-debt categories associated with the three scenarios provided in the assignment.

The scenarios involve:

* Changes to one feature unexpectedly affecting an unrelated feature.
* Another team silently depending on raw model outputs.
* A training pipeline consisting of undocumented shell scripts without an orchestration tool.

The question also requires proposing a specific mitigation for one of the three cases.

## Answer

The completed answer is available in:

```text
Q1_Technical_Debt/answer.md
```

No code execution is required for Q1.

---

# Question 2 — MLflow Experiment Comparison

## Objective

Q2 changes the original training setup by:

* Replacing the Random Forest predictor with an MLP.
* Changing the dataset from IRIS to MNIST.
* Running at least six experiments.
* Varying at least two hyperparameters.
* Tracking the experiments using MLflow.

The assignment requires a comparison of all six runs and a written analysis of the results.

---

## Q2 — Step 1: Start MLflow

MLflow is started manually before running the training script.

Open **Terminal 1**.

Activate the environment:

```bash
mamba activate aiops
```

Navigate to the repository:

```bash
cd ~/AIOps-Module1
```

Start the MLflow server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Keep this terminal running.

The MLflow UI will be available at:

```text
http://127.0.0.1:5000
```

Open the address in your browser.

---

## Q2 — Step 2: Run All Six Experiments

Open **Terminal 2**.

Activate the environment:

```bash
mamba activate aiops
```

Navigate to the Q2 directory:

```bash
cd ~/AIOps-Module1/Q2_MLflow
```

Run:

```bash
python train_mnist_mlp.py
```

### Important

**Only one training command is required.**

The `train_mnist_mlp.py` script automatically executes all six experiment configurations.

You do **not** need to run the training script six separate times.

Each configuration is recorded as a separate MLflow run.

The experiments log the relevant hyperparameters and performance metrics.

---

## Q2 — Step 3: View the MLflow Runs

Return to the browser:

```text
http://127.0.0.1:5000
```

Open the MLflow experiment containing the Q2 runs.

The six runs can be compared using the MLflow comparison interface.

The comparison includes the logged parameters and metrics for each experiment.

---

## Q2 — Step 4: Compare the Six Experiments

The six runs are compared to identify the best-performing configuration.

The MLflow comparison screenshot is located at:

```text
Q2_MLflow/results/mlflow_comparision.png
```

---

## Q2 — Step 5: Experiment Analysis

The written analysis is available at:

```text
Q2_MLflow/analysis.md
```

The analysis covers:

1. The best-performing run and why it performed best.
2. Evidence of overfitting from the training-loss and validation-accuracy trend.
3. Which hyperparameter appears to have the larger effect on performance.

The assignment requires a **150–250 word analysis**.

---

## Q2 — Quick Reference

### Terminal 1 — MLflow

```bash
mamba activate aiops
cd ~/AIOps-Module1

mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Keep this terminal running.

Open:

```text
http://127.0.0.1:5000
```

### Terminal 2 — Training

```bash
mamba activate aiops
cd ~/AIOps-Module1/Q2_MLflow

python train_mnist_mlp.py
```

This single command automatically trains **all six experiments**.

---

# Question 3 — DVC Data Versioning & Rollback

## Objective

Q3 demonstrates dataset versioning and rollback using **DVC and Git**.

The initial dataset is used to create a CSV containing the filenames.

The required versions are:

| Version | Data Rows | CSV Lines Including Header |
| ------- | --------: | -------------------------: |
| v1      |      1800 |                       1801 |
| v2      |      2801 |                       2802 |

The assignment requires creating v1, updating the dataset to v2, and then demonstrating a rollback to v1.

---

## Q3 — Version 1

The initial dataset is tracked using DVC.

The DVC metadata is stored in:

```text
Q3_DVC/data.dvc
```

The v1 CSV contains:

```text
1800 data rows
+ 1 header
= 1801 lines
```

Verify the row count:

```bash
wc -l Q3_DVC/file_list.csv
```

Expected result:

```text
1801
```

---

## Q3 — Version 2

The dataset is updated using the new labels.

The v2 dataset contains:

```text
2801 data rows
+ 1 header
= 2802 lines
```

The updated dataset is tracked using DVC and the corresponding metadata is committed to Git.

---

## Q3 — Rollback to v1

To restore the v1 dataset:

```bash
git checkout v1
dvc checkout
```

Then verify the CSV:

```bash
wc -l Q3_DVC/file_list.csv
```

Expected result:

```text
1801
```

This confirms:

```text
1800 data rows + 1 header
```

The rollback evidence is documented in:

```text
Q3_DVC/rollback.md
```

The assignment specifically requires evidence proving that the row count after rollback matches v1 exactly.

---

## Q3 — Return to v2

To return to the latest version:

```bash
git checkout main
dvc checkout
```

The v2 CSV should contain:

```text
2802 lines
```

corresponding to:

```text
2801 data rows + 1 header
```

---

# Question 4 — End-to-End Reproducibility Drill

## Objective

Q4 demonstrates an end-to-end reproducibility workflow between Partner A and Partner B.

The workflow combines:

```text
Git
+
DVC
+
MLflow
+
Environment Management
+
Model Training
```

Partner A performs the original experiment, versions the dataset, logs the run in MLflow, and commits the required code and DVC metadata.

Partner B then reproduces the result using the repository, Git commit, DVC checkout, and environment definition.

---

## Partner A

Partner A trains the model and logs the experiment in MLflow.

The run records:

* Model parameters
* Metrics
* Random seed
* Git commit
* Model artifact

The dataset is versioned using DVC.

The code and `.dvc` file are committed to Git.

The model is registered and transitioned to:

```text
Staging
```

---

## Partner B — Reproduction

Partner B performs the reproduction using the following repository:

**Partner B Repository**

```text
https://github.com/kirthans/da3408-mnist-mlp-repro.git
```

### Step 1 — Clone the Partner B Repository

```bash
git clone https://github.com/kirthans/da3408-mnist-mlp-repro.git
cd da3408-mnist-mlp-repro
```

### Step 2 — Check Out the Required Commit

```bash
git checkout <commit-hash>
```

### Step 3 — Restore the DVC Dataset

```bash
dvc checkout
```

### Step 4 — Create the Environment

```bash
mamba env create -f environment.yml
```

### Step 5 — Activate the Environment

```bash
mamba activate aiops
```

### Step 6 — Run the Training Script

```bash
python train.py
```

The exact training command should follow the training script in the Partner B repository.

---

## Reproduction Result

Partner B compares the reproduced metric with Partner A's original metric.

The comparison includes:

* Original metric
* Reproduced metric
* Metric difference
* Stated tolerance
* Whether the result matched
* Explanation of any discrepancy

The reproduction result is documented in:

```text
Q4_Reproducibility/reproduction.md
```

Partner B also records a note in the MLflow run documenting whether the metric matched within the stated tolerance or explaining any discrepancy.

---

## Q4 Partner B Repository

The repository used for the Partner B reproduction is:

```text
https://github.com/kirthans/da3408-mnist-mlp-repro.git
```

---

# Reproducibility Workflow

The overall workflow is:

```text
                 ┌─────────────────┐
                 │      Git        │
                 │ Code + Versions │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │      DVC        │
                 │ Dataset Version │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │     Training    │
                 │      MLP        │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │     MLflow      │
                 │ Params/Metrics  │
                 │ Artifacts/Model │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    Partner B    │
                 │   Reproduction  │
                 └─────────────────┘
```

---

# Tools Used

| Tool              | Purpose                                                                   |
| ----------------- | ------------------------------------------------------------------------- |
| **Git**           | Source-code and version control                                           |
| **DVC**           | Dataset versioning and rollback                                           |
| **MLflow**        | Experiment tracking, parameters, metrics, artifacts, and model management |
| **Mamba / Conda** | Reproducible environment creation                                         |
| **Python**        | Model training and experiment execution                                   |
| **scikit-learn**  | Machine-learning utilities and evaluation                                 |
| **pandas**        | Data processing                                                           |

---

# Assignment Deliverables

## Q1 — Technical Debt Diagnosis

```text
Q1_Technical_Debt/answer.md
```

Contains:

* Technical-debt classification for the three scenarios.
* Mitigation for one scenario.

---

## Q2 — MLflow Experiment Comparison

```text
Q2_MLflow/
├── train_mnist_mlp.py
├── analysis.md
└── results/
    └── mlflow_comparision.png
```

Contains:

* Six MLflow experiment runs.
* MLflow comparison screenshot.
* Experiment analysis.
* MLflow parameter and metric logging code.

---

## Q3 — DVC Data Versioning & Rollback

```text
Q3_DVC/
├── data.dvc
├── file_list.csv
└── rollback.md
```

Contains:

* v1 dataset version.
* v2 dataset version.
* DVC tracking.
* Git commits.
* Rollback to v1.
* Row-count evidence.

---

## Q4 — End-to-End Reproducibility

```text
Q4_Reproducibility/
├── train.py
└── reproduction.md
```

Contains:

* Partner A training result.
* MLflow tracking.
* DVC dataset versioning.
* Git commit.
* Partner B reproduction.
* Metric comparison.
* Tolerance/discrepancy documentation.

---

# Quick Start

## Clone and set up

```bash
git clone https://github.com/Ramanagopalkodali/AIOps-Module1.git
cd AIOps-Module1

mamba env create -f environment.yml
mamba activate aiops

pip install "dvc[ssh]"
```

---

## Q1

Read:

```bash
cat Q1_Technical_Debt/answer.md
```

---

## Q2

### Terminal 1 — Start MLflow

```bash
mamba activate aiops
cd ~/AIOps-Module1

mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Keep Terminal 1 running.

### Terminal 2 — Run all six experiments

```bash
mamba activate aiops
cd ~/AIOps-Module1/Q2_MLflow

python train_mnist_mlp.py
```

The single command automatically runs all six experiments.

---

## Q3

Restore v1:

```bash
git checkout v1
dvc checkout
```

Check the row count:

```bash
wc -l Q3_DVC/file_list.csv
```

Expected:

```text
1801
```

---

## Q4

Use the Partner B repository:

```bash
git clone https://github.com/kirthans/da3408-mnist-mlp-repro.git
cd da3408-mnist-mlp-repro
```

Then:

```bash
git checkout <commit-hash>
dvc checkout
mamba env create -f environment.yml
mamba activate aiops
python train.py
```

Document the reproduction result in the appropriate Q4 evidence.

---

# Repositories

## Main Repository

https://github.com/Ramanagopalkodali/AIOps-Module1

## Q4 Partner B Reproduction Repository

https://github.com/kirthans/da3408-mnist-mlp-repro.git

---

# Conclusion

This project demonstrates an end-to-end approach to **ML experiment management and reproducibility**.

The workflow integrates:

```text
Git
↓
DVC
↓
MLflow
↓
Environment Management
↓
Reproducible Model Training
```

Together, these tools provide versioned source code, versioned datasets, tracked experiments, reproducible environments, and documented model-training results.
