from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import DATA_PATH, MODELS_DIR
from src.pipeline import clean_dataset, load_dataset


MODEL_PATH = MODELS_DIR / "best_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python3 -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


def expected_columns(model) -> tuple[list[str], list[str]]:
    preprocessor = model.named_steps["preprocessor"]
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for name, _, columns in preprocessor.transformers_:
        if name == "num":
            numeric_columns = list(columns)
        elif name == "cat":
            categorical_columns = list(columns)

    return numeric_columns, categorical_columns


def load_reference_options() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = load_dataset(DATA_PATH)
    X, _ = clean_dataset(df)
    options: dict[str, list[str]] = {}

    for column in X.columns:
        series = X[column].dropna()
        if series.empty:
            continue
        if pd.api.types.is_numeric_dtype(series):
            unique_values = sorted(series.astype(str).unique().tolist())
        else:
            unique_values = sorted(series.astype(str).str.strip().unique().tolist())
        options[column] = unique_values[:8]

    return X, options


def prompt_numeric(column: str, samples: list[str], index: int, total: int):
    hint = ""
    if samples:
        hint = f" Example values: {', '.join(samples)}"
    while True:
        raw = input(f"[{index}/{total}] {column}{hint}\n> ").strip()
        if raw == "":
            return None
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            print("Enter a numeric value or leave it blank.")


def prompt_categorical(column: str, samples: list[str], index: int, total: int):
    hint = ""
    if samples:
        hint = f" Example values: {', '.join(samples)}"
    raw = input(f"[{index}/{total}] {column}{hint}\n> ").strip()
    return raw or None


def collect_answers(model) -> pd.DataFrame:
    _, options = load_reference_options()
    numeric_columns, categorical_columns = expected_columns(model)
    all_columns = numeric_columns + categorical_columns
    answers: dict[str, object] = {}

    print("\nMental Health Treatment Prediction Questionnaire")
    print("Leave a question blank if you do not know the answer.\n")

    for index, column in enumerate(all_columns, start=1):
        sample_values = options.get(column, [])
        if column in numeric_columns:
            answers[column] = prompt_numeric(column, sample_values, index, len(all_columns))
        else:
            answers[column] = prompt_categorical(
                column, sample_values, index, len(all_columns)
            )

    return pd.DataFrame([answers], columns=all_columns)


def show_prediction(model, answers: pd.DataFrame) -> None:
    prediction = int(model.predict(answers)[0])
    probability = float(model.predict_proba(answers)[0, 1])

    label = "Likely sought treatment" if prediction == 1 else "Likely did not seek treatment"
    print("\nPrediction Result")
    print(f"Predicted class: {prediction}")
    print(f"Interpretation: {label}")
    print(f"Probability of seeking treatment: {probability:.2%}")


def main() -> None:
    model = load_model()
    answers = collect_answers(model)
    show_prediction(model, answers)


if __name__ == "__main__":
    main()
