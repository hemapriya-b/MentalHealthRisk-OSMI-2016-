# OSMI 2016 Mental Health Treatment Prediction

This project builds a leak-safe machine learning pipeline on the OSMI 2016 workplace mental health survey dataset to predict whether a respondent has sought treatment from a mental health professional.

## Project Goal

Predict the target column:

`Have you ever sought treatment for a mental health issue from a mental health professional?`

The project focuses on:

- clean preprocessing for mixed survey data
- leak-safe train/test handling
- feature selection
- class imbalance handling
- model comparison and tuning
- explainable output for reporting

## Dataset

- File: `data/raw/dataset.csv`
- Rows: `1433`
- Columns: `63`
- Data type mix: categorical, ordinal-style survey responses, and numerical fields such as age

## Final Verified Results

Best verified leak-safe model:

- Model: `RandomForest`
- Accuracy: `85.37%`
- Precision: `0.8795`
- Recall: `0.8690`
- F1-score: `0.8743`
- ROC-AUC: `0.9274`

These results are stored in:

- `reports/metrics.json`
- `reports/training_summary.json`

## Project Structure

```text
MR 3-4/
├── data/
│   └── raw/
│       └── dataset.csv
├── models/
│   └── best_model.joblib
├── reports/
│   ├── confusion_matrix.png
│   ├── metrics.json
│   ├── selected_features.txt
│   ├── shap_feature_importance.csv
│   └── training_summary.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── pipeline.py
│   └── train.py
└── requirements.txt
```

## Final Pipeline

### 1. Data Cleaning

- loads the raw dataset
- normalizes column names
- extracts the binary target
- removes free-text response columns
- cleans age and basic gender values

### 2. Leak-Safe Train/Test Handling

- splits the dataset before train-derived filtering
- identifies high-missing columns using `X_train` only
- applies the same training-derived drop list to `X_test`

This is important because it prevents the test set from influencing preprocessing decisions.

### 3. Preprocessing

- median imputation for numeric features
- most-frequent imputation for categorical features
- one-hot encoding for categorical features
- min-max scaling for numeric features

### 4. Feature Selection

- `SelectKBest(chi2)` after preprocessing
- selected transformed features are exported to `reports/selected_features.txt`

### 5. Class Balancing

- `SMOTETomek` is applied inside the training pipeline
- because it happens after the split and inside the pipeline, there is no leakage from resampling

### 6. Model Candidates

The project compares and tunes these models:

- Logistic Regression
- Random Forest
- Extra Trees
- Naive Bayes
- CatBoost
- XGBoost

The saved best model is chosen by tuned cross-validation on the training split, not by selecting the highest test-set score after the fact.

## Best Final Parameters

Final tuned Random Forest parameters from `training_summary.json`:

- `n_estimators = 700`
- `min_samples_split = 4`
- `min_samples_leaf = 4`
- `max_features = "log2"`
- `max_depth = 12`
- `feature_selector__k = 80`

## Why This Version Was Kept

Several more aggressive experiments were tested, including stricter proxy-reduction, broader imputation ideas, feature-engineering attempts, and more complex hybrid setups.

Those variants were rejected because they reduced real holdout performance.

The current saved version is the strongest verified configuration that remained technically leak-safe.

## Latest Comparison Snapshot

From the latest training run:

- `RandomForest` remained the saved model because it achieved the strongest tuned cross-validation score on the training split.
- `ExtraTrees`, `LogisticRegression`, and `NaiveBayes` each reached `85.71%` holdout accuracy in the side-by-side comparison.
- `ExtraTrees` produced the strongest holdout ROC-AUC at roughly `0.9300`.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Training

```bash
python3 -m src.train
```

## Run Interactive Prediction

If you want to answer questions manually and get a prediction from the saved model:

```bash
python3 test.py
```

The script will ask the required survey-style questions one by one, then print:

- predicted class
- plain-language interpretation
- probability of seeking treatment

## Generated Outputs

After training, the project writes:

- `models/best_model.joblib`
- `reports/metrics.json`
- `reports/model_comparison.csv`
- `reports/model_comparison.json`
- `reports/training_summary.json`
- `reports/confusion_matrix.png`
- `reports/selected_features.txt`
- `reports/shap_feature_importance.csv`

## Academic Note

This pipeline is designed to be technically leak-safe:

- preprocessing is fit inside the pipeline
- train-derived filtering is computed from the training split only
- tuning is performed on training data only
- resampling is performed inside the pipeline

There may still be domain-level discussion about whether some survey questions are conceptually close to the target, but the workflow itself does not show obvious technical data leakage.

## Summary

This project provides a strong, honest, and reproducible machine learning workflow for predicting treatment-seeking behavior using the OSMI 2016 mental health survey dataset.

It did not reach the original 90% stretch goal, but it produced a strong final result with:

- solid generalization
- high ROC-AUC
- interpretable outputs
- leak-safe methodology suitable for academic presentation
# MentalHealthRisk-OSMI-2016-
