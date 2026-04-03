from __future__ import annotations

import json

from sklearn.model_selection import train_test_split

from .config import DATA_PATH, REPORTS_DIR, RANDOM_STATE, TEST_SIZE
from .pipeline import (
    apply_train_based_column_filter,
    clean_dataset,
    cross_validate_models,
    fit_and_evaluate,
    load_dataset,
    tune_top_models,
)


def main() -> None:
    df = load_dataset(DATA_PATH)
    X, y = clean_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_test, dropped_columns = apply_train_based_column_filter(X_train, X_test)

    best_model_name, cv_scores = cross_validate_models(X_train, y_train)
    tuned_model_name, tuned_pipeline, tuning_results = tune_top_models(
        X_train, y_train, cv_scores
    )
    artifacts = fit_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        tuned_model_name,
        tuned_pipeline,
        tuning_results,
    )

    summary = {
        "dataset_shape": df.shape,
        "feature_shape_after_cleaning": X.shape,
        "train_shape": artifacts.train_shape,
        "test_shape": artifacts.test_shape,
        "baseline_best_model": best_model_name,
        "best_model": artifacts.best_model_name,
        "cross_validation_accuracy": cv_scores,
        "tuning_results": artifacts.tuning_results,
        "holdout_metrics": artifacts.metrics,
        "selected_feature_count": len(artifacts.selected_features),
        "dropped_columns_from_train_missingness": dropped_columns,
        "target_distribution": {
            "negative_class": int((y == 0).sum()),
            "positive_class": int((y == 1).sum()),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
