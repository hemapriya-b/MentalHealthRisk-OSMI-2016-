from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent / ".matplotlib"),
)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .config import (
    AGE_COLUMN,
    CV_FOLDS,
    GENDER_COLUMN,
    HIGH_MISSING_DROP_THRESHOLD,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEXT_HEAVY_COLUMNS,
    TUNING_ITERATIONS,
)


@dataclass
class TrainingArtifacts:
    best_model_name: str
    metrics: dict[str, Any]
    model_comparison: dict[str, dict[str, Any]]
    selected_features: list[str]
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    tuning_results: dict[str, Any]


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = [str(column).replace("\xa0", " ").strip() for column in df.columns]
    return df


def normalize_gender(value: Any) -> str:
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()
    if not text:
        return "unknown"
    if any(token in text for token in ("female", "woman", "f ")):
        return "female"
    if any(token in text for token in ("male", "man", "m ")):
        return "male"
    if any(token in text for token in ("non-binary", "nonbinary", "genderqueer", "trans")):
        return "non_binary_or_trans"
    return "other"


def encode_target(series: pd.Series) -> pd.Series:
    mapping = {
        "1": 1,
        "0": 0,
        1: 1,
        0: 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
    }
    cleaned = series.astype(str).str.strip().str.lower().map(mapping)
    if cleaned.isna().any():
        missing_values = sorted(series[cleaned.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unexpected target values found: {missing_values}")
    return cleaned.astype(int)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column not found: {TARGET_COLUMN}")

    df = df.copy()
    y = encode_target(df.pop(TARGET_COLUMN))

    df = df.drop(columns=TEXT_HEAVY_COLUMNS, errors="ignore")
    df = basic_clean_features(df)

    return df, y


def basic_clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if AGE_COLUMN in df.columns:
        df[AGE_COLUMN] = pd.to_numeric(df[AGE_COLUMN], errors="coerce")
        df.loc[(df[AGE_COLUMN] < 18) | (df[AGE_COLUMN] > 80), AGE_COLUMN] = np.nan
    if GENDER_COLUMN in df.columns:
        df[GENDER_COLUMN] = df[GENDER_COLUMN].map(normalize_gender)

    return df


def columns_to_drop_from_train(X_train: pd.DataFrame) -> list[str]:
    missing_ratio = X_train.isna().mean()
    return missing_ratio[missing_ratio > HIGH_MISSING_DROP_THRESHOLD].index.tolist()


def apply_train_based_column_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    drop_columns = columns_to_drop_from_train(X_train)
    filtered_train = X_train.drop(columns=drop_columns, errors="ignore")
    filtered_test = X_test.drop(columns=drop_columns, errors="ignore")
    return filtered_train, filtered_test, drop_columns


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if AGE_COLUMN in df.columns:
        age = pd.to_numeric(df[AGE_COLUMN], errors="coerce")
        df["engineered_age_bucket"] = pd.cut(
            age,
            bins=[17, 25, 35, 50, 80],
            labels=["18_25", "26_35", "36_50", "51_80"],
        ).astype("object")

    return df


def make_one_hot_encoder() -> OneHotEncoder:
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def choose_k_features(X: pd.DataFrame) -> int | str:
    if X.shape[1] <= 15:
        return "all"  # type: ignore[return-value]
    return min(80, max(20, X.shape[1] * 2))


def build_candidate_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "naive_bayes": GaussianNB(),
    }

    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=False,
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        pass

    return models


def build_training_pipeline(X: pd.DataFrame, estimator: Any) -> ImbPipeline:
    preprocessor, _, _ = build_preprocessor(X)
    k = choose_k_features(X)
    return ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selector", SelectKBest(score_func=chi2, k=k)),
            ("resample", SMOTETomek(random_state=RANDOM_STATE)),
            ("model", estimator),
        ]
    )


def cross_validate_models(X: pd.DataFrame, y: pd.Series) -> tuple[str, dict[str, float]]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores: dict[str, float] = {}
    for name, estimator in build_candidate_models().items():
        pipeline = build_training_pipeline(X, estimator)
        score = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        ).mean()
        scores[name] = float(score)
    best_model_name = max(scores, key=scores.get)
    return best_model_name, scores


def get_param_distributions(model_name: str) -> dict[str, list[Any]]:
    distributions: dict[str, dict[str, list[Any]]] = {
        "logistic_regression": {
            "feature_selector__k": [40, 60, 80, "all"],
            "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "model__solver": ["lbfgs"],
        },
        "random_forest": {
            "feature_selector__k": [40, 60, 80, "all"],
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [None, 8, 12, 18],
            "model__min_samples_split": [2, 4, 8],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
        "extra_trees": {
            "feature_selector__k": [40, 60, 80, "all"],
            "model__n_estimators": [300, 500, 700, 900],
            "model__max_depth": [None, 8, 12, 18],
            "model__min_samples_split": [2, 4, 8],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
        "naive_bayes": {
            "feature_selector__k": [20, 40, 60, 80, "all"],
            "model__var_smoothing": np.logspace(-11, -7, 9).tolist(),
        },
        "catboost": {
            "feature_selector__k": [40, 60, 80, "all"],
            "model__iterations": [200, 400, 600],
            "model__depth": [4, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__l2_leaf_reg": [1.0, 3.0, 5.0, 7.0],
        },
        "xgboost": {
            "feature_selector__k": [40, 60, 80, "all"],
            "model__n_estimators": [200, 350, 500],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.10],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0],
        },
    }
    return distributions.get(model_name, {})


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_scores: dict[str, float],
) -> tuple[str, dict[str, ImbPipeline], dict[str, Any]]:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    models = build_candidate_models()
    ranked_names = sorted(cv_scores, key=cv_scores.get, reverse=True)

    best_name = ranked_names[0]
    best_score = cv_scores[best_name]
    tuned_pipelines: dict[str, ImbPipeline] = {}
    tuning_results: dict[str, Any] = {}

    for model_name in ranked_names:
        estimator = build_training_pipeline(X_train, models[model_name])
        distributions = get_param_distributions(model_name)

        if distributions:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=distributions,
                n_iter=min(TUNING_ITERATIONS, np.prod([len(v) for v in distributions.values()])),
                scoring="accuracy",
                cv=cv,
                n_jobs=1,
                random_state=RANDOM_STATE,
                refit=True,
            )
            search.fit(X_train, y_train)
            score = float(search.best_score_)
            tuned_pipelines[model_name] = search.best_estimator_
            tuning_results[model_name] = {
                "best_cv_accuracy": score,
                "best_params": search.best_params_,
            }
            if score > best_score:
                best_score = score
                best_name = model_name
        else:
            tuned_pipelines[model_name] = estimator
            tuning_results[model_name] = {
                "best_cv_accuracy": float(cv_scores[model_name]),
                "best_params": {},
            }

    return best_name, tuned_pipelines, tuning_results


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def fit_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_model_name: str,
    candidate_pipelines: dict[str, ImbPipeline],
    cv_scores: dict[str, float],
    tuning_results: dict[str, Any],
) -> TrainingArtifacts:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_comparison: dict[str, dict[str, Any]] = {}
    best_pipeline: ImbPipeline | None = None
    best_metrics: dict[str, Any] = {}
    best_feature_names: list[str] = []
    best_predictions: np.ndarray | None = None

    for model_name, pipeline in candidate_pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        model_comparison[model_name] = {
            "baseline_cv_accuracy": float(cv_scores[model_name]),
            "tuned_cv_accuracy": float(tuning_results[model_name]["best_cv_accuracy"]),
            **metrics,
        }

        if model_name == best_model_name:
            best_pipeline = pipeline
            best_metrics = metrics
            best_feature_names = extract_selected_feature_names(pipeline)
            best_predictions = y_pred

    if best_pipeline is None or best_predictions is None:
        raise ValueError(f"Best model pipeline not found for {best_model_name}")

    comparison_df = (
        pd.DataFrame.from_dict(model_comparison, orient="index")
        .reset_index()
        .rename(columns={"index": "model"})
        .sort_values(["accuracy", "roc_auc"], ascending=[False, False])
    )
    comparison_export_df = comparison_df.drop(columns=["classification_report"])
    comparison_export_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    (REPORTS_DIR / "model_comparison.json").write_text(
        comparison_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)

    scores_path = REPORTS_DIR / "metrics.json"
    with scores_path.open("w", encoding="utf-8") as file_obj:
        json.dump(best_metrics, file_obj, indent=2)

    try:
        plot_confusion_matrix(y_test, best_predictions, REPORTS_DIR / "confusion_matrix.png")
    except Exception:
        pass

    feature_path = REPORTS_DIR / "selected_features.txt"
    feature_path.write_text("\n".join(best_feature_names), encoding="utf-8")
    try:
        export_shap_summary(best_pipeline, X_train, REPORTS_DIR)
    except Exception:
        pass

    return TrainingArtifacts(
        best_model_name=best_model_name,
        metrics=best_metrics,
        model_comparison=model_comparison,
        selected_features=best_feature_names,
        train_shape=X_train.shape,
        test_shape=X_test.shape,
        tuning_results=tuning_results,
    )


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def extract_selected_feature_names(pipeline: ImbPipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    selector = pipeline.named_steps["feature_selector"]
    feature_names = preprocessor.get_feature_names_out()
    support = selector.get_support()
    return feature_names[support].tolist()


def export_shap_summary(
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    output_dir: Path,
) -> None:
    import shap

    transformed = pipeline.named_steps["preprocessor"].transform(X_train)
    selected = pipeline.named_steps["feature_selector"].transform(transformed)
    feature_names = extract_selected_feature_names(pipeline)
    model = pipeline.named_steps["model"]

    sample_size = min(200, selected.shape[0])
    sample = selected[:sample_size]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
    except Exception:
        return

    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[-1])
    else:
        shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, -1]

    mean_abs = np.abs(shap_array).mean(axis=0)
    importance = (
        pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    plt.figure(figsize=(8, 6))
    top_n = importance.head(15).iloc[::-1]
    plt.barh(top_n["feature"], top_n["mean_abs_shap"], color="#2c7fb8")
    plt.title("Top SHAP Feature Importance")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_feature_importance.png", dpi=200)
    plt.close()
