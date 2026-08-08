# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-1893F8)
![Use](https://img.shields.io/badge/Use-Educational-31C48D)

An end-to-end machine-learning study for detecting fraudulent credit-card
transactions in a highly imbalanced dataset. The project compares four
classifiers across three resampling strategies and evaluates them with metrics
that reflect the real cost of fraud detection: precision, recall, F1-score, and
ROC-AUC.

## Project highlights

- Explores 284,807 anonymized European card transactions.
- Handles a fraud rate of only **0.173%** (492 fraudulent transactions).
- Compares Random Undersampling, SMOTE, and SMOTEENN.
- Benchmarks Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- Uses a stratified train/test split and fits preprocessing on training data only.
- Selects the best experiment by F1-score instead of misleading raw accuracy.

## Best result

The strongest experiment in the notebook is **Random Forest with SMOTEENN**.

| Metric | Score |
|---|---:|
| Precision | 0.8736 |
| Recall | 0.8000 |
| F1-score | **0.8352** |
| ROC-AUC | 0.9655 |

The highest ROC-AUC, **0.9766**, was produced by XGBoost with random
undersampling, but its low precision made it less suitable as the overall winner.

## Experiment results

| Rank | Sampling | Model | Precision | Recall | F1-score | ROC-AUC |
|---:|---|---|---:|---:|---:|---:|
| 1 | SMOTEENN | Random Forest | 0.8736 | 0.8000 | **0.8352** | 0.9655 |
| 2 | SMOTE | Random Forest | **0.9125** | 0.7684 | 0.8343 | 0.9545 |
| 3 | SMOTE | XGBoost | 0.7475 | 0.7789 | 0.7629 | 0.9634 |
| 4 | SMOTEENN | XGBoost | 0.7196 | 0.8105 | 0.7624 | 0.9637 |
| 5 | SMOTEENN | Decision Tree | 0.3613 | 0.7263 | 0.4825 | 0.8621 |
| 6 | SMOTE | Decision Tree | 0.3516 | 0.6737 | 0.4621 | 0.8358 |
| 7 | Undersampling | Random Forest | 0.0792 | 0.8632 | 0.1450 | 0.9746 |
| 8 | SMOTE | Logistic Regression | 0.0536 | 0.8737 | 0.1010 | 0.9595 |
| 9 | SMOTEENN | Logistic Regression | 0.0527 | 0.8737 | 0.0994 | 0.9596 |
| 10 | Undersampling | XGBoost | 0.0510 | 0.8632 | 0.0963 | **0.9766** |
| 11 | Undersampling | Logistic Regression | 0.0501 | 0.8737 | 0.0948 | 0.9547 |
| 12 | Undersampling | Decision Tree | 0.0190 | **0.9158** | 0.0373 | 0.9183 |

## Repository structure

```text
.
├── Fraud Detection.ipynb   # EDA, preprocessing, training, and evaluation
├── requirements.txt        # Reproducible notebook dependencies
├── .gitignore
└── README.md
```

The dataset is intentionally excluded because the CSV is larger than GitHub's
standard file-size limit. Local application files are also excluded from this
repository.

## Dataset

This project uses the
[Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
It contains transactions made by European cardholders during two days in
September 2013.

Most input variables (`V1`–`V28`) are PCA-transformed for confidentiality.
`Time` and `Amount` retain their original meanings, and `Class` is the target:

- `0` — legitimate transaction
- `1` — fraudulent transaction

Download `creditcard.csv` from Kaggle and place it in the repository root before
running the notebook.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Nour-Elrouby/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset and launch Jupyter

```bash
jupyter notebook "Fraud Detection.ipynb"
```

## Methodology

1. **Data audit** — inspect schema, class distribution, missing values, and duplicates.
2. **Exploratory analysis** — visualize class balance, correlations, and selected feature relationships.
3. **Preprocessing** — remove duplicate rows and the `Time` column, then robust-scale `Amount`.
4. **Stratified split** — preserve the rare fraud ratio in the held-out test set.
5. **Resampling** — compare undersampling, SMOTE, and SMOTEENN on training data.
6. **Modeling** — train four classification algorithms for every sampling strategy.
7. **Evaluation** — compare confusion matrices, ROC curves, precision, recall, F1, and ROC-AUC.

## Key takeaways

- Accuracy is not informative when more than 99.8% of transactions are legitimate.
- Random Forest delivered the best F1-score after hybrid SMOTEENN resampling.
- Undersampling increased recall and ROC-AUC in several experiments, but produced
  too many false positives and therefore very low precision.
- The operating threshold should ultimately be tuned to the business cost of a
  missed fraud case versus a false alert.

## Future improvements

- Tune model hyperparameters with stratified cross-validation.
- Optimize the probability threshold against a cost-sensitive objective.
- Compare class-weighted learners with resampling approaches.
- Add probability calibration and precision-recall curves.
- Use time-aware validation to better approximate production deployment.
- Package preprocessing and inference into a versioned model pipeline.

## Responsible use

This repository is an educational machine-learning project. A fraud score should
support human review and layered risk controls; it should not be the sole basis
for blocking transactions or making decisions about customers.
