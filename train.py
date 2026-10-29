from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = int(os.getenv("CARWISE_SEED", "42"))
TEST_SIZE = 0.2
SOURCE_PATH = Path(os.getenv("CARWISE_DATASET", "data/original.csv"))
CLEANED_PATH = Path("data/cleaned.csv")
ARTIFACT_DIR = Path("artifacts")
TARGET = "selling_price"
FEATURES = ["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner"]
NUMERIC_FEATURES = ["year", "km_driven"]
CATEGORICAL_FEATURES = ["name", "fuel", "seller_type", "transmission", "owner"]
MULTI_WORD_BRANDS = ("Land Rover",)


def canonicalize(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_category(value: Any, field: str) -> Any:
    value = canonicalize(value)
    if pd.isna(value):
        return np.nan
    key = str(value).lower()
    mappings = {
        "fuel": {"cng": "CNG", "diesel": "Diesel", "lpg": "LPG", "petrol": "Petrol"},
        "seller_type": {"dealer": "Dealer", "individual": "Individual", "trustmark dealer": "Trustmark Dealer"},
        "transmission": {"automatic": "Automatic", "manual": "Manual"},
        "owner": {
            "first owner": "First Owner",
            "second owner": "Second Owner",
            "third owner": "Third Owner",
            "fourth & above owner": "Fourth & Above Owner",
            "test drive car": "Test Drive Car",
        },
    }
    return mappings.get(field, {}).get(key, value)


def extract_brand(name: str) -> str:
    """Derive a display brand from the existing dataset name without changing model inputs."""
    for brand in MULTI_WORD_BRANDS:
        if name == brand or name.startswith(f"{brand} "):
            return brand
    return name.split(" ", 1)[0]


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    before_rows = len(raw)
    expected = FEATURES + [TARGET]
    missing_columns = [column for column in expected if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    frame = raw[expected].copy()
    duplicate_rows = int(frame.duplicated().sum())
    missing_before = {column: int(frame[column].isna().sum()) for column in expected}

    frame["name"] = frame["name"].map(canonicalize)
    for field in ["fuel", "seller_type", "transmission", "owner"]:
        frame[field] = frame[field].map(lambda value, field=field: normalize_category(value, field))
    for field in NUMERIC_FEATURES + [TARGET]:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")

    invalid_mask = (
        (frame["year"].notna() & ~frame["year"].between(1980, 2026))
        | (frame["km_driven"].notna() & ~frame["km_driven"].between(0, 1_000_000))
        | (frame[TARGET].notna() & ~frame[TARGET].between(1_000, 20_000_000))
    )
    invalid_rows = int(invalid_mask.sum())
    frame = frame.loc[~invalid_mask].dropna(subset=expected).drop_duplicates().reset_index(drop=True)

    for field in NUMERIC_FEATURES + [TARGET]:
        if field in frame:
            frame[field] = frame[field].astype(int)

    cleaning = {
        "rows_before": before_rows,
        "rows_after": len(frame),
        "duplicate_rows_removed": duplicate_rows,
        "invalid_rows_removed": invalid_rows,
        "missing_values_before": missing_before,
        "missing_values_after": {column: int(frame[column].isna().sum()) for column in expected},
        "decisions": [
            "Selected only the eight supported source columns.",
            "Trimmed and collapsed repeated whitespace in text fields.",
            "Canonicalized known categorical spellings without inventing values.",
            "Coerced numeric columns to numbers and removed malformed or unrealistic rows.",
            "Removed exact duplicate records after measuring them; no near-duplicates were inferred.",
            "Dropped incomplete rows because all model inputs are required by the assignment and the source contains enough complete records.",
        ],
    }
    return frame, cleaning


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def build_models() -> dict[str, Any]:
    return {
        "Dummy Regressor": DummyRegressor(strategy="median"),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=180, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.05, max_depth=3, loss="huber", random_state=SEED
        ),
    }


def evaluate_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[Pipeline, dict[str, float]]:
    pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", model)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }
    return pipeline, metrics


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Could not find source dataset at {SOURCE_PATH}")

    raw = pd.read_csv(SOURCE_PATH)
    cleaned, cleaning = clean_dataset(raw)
    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEANED_PATH, index=False)

    X = cleaned[FEATURES]
    y = cleaned[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    comparisons: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, model in build_models().items():
        pipeline, metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        fitted[name] = pipeline
        comparisons.append({"model": name, **metrics})

    # RMSE is the primary selection metric because it penalizes large resale-price errors;
    # R² is reported alongside it to make explanatory power visible.
    comparisons.sort(key=lambda item: item["rmse"])
    selected_name = comparisons[0]["model"]
    selected_pipeline = fitted[selected_name]

    cv_pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", build_models()[selected_name])])
    cv = cross_validate(
        cv_pipeline,
        X_train,
        y_train,
        cv=KFold(n_splits=3, shuffle=True, random_state=SEED),
        scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
        n_jobs=1,
    )
    cv_metrics = {
        "mae_mean": float(-cv["test_mae"].mean()),
        "mae_std": float(cv["test_mae"].std()),
        "rmse_mean": float(-cv["test_rmse"].mean()),
        "r2_mean": float(cv["test_r2"].mean()),
    }

    # After held-out selection, refit the exact selected pipeline on all cleaned rows for production inference.
    production_pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", build_models()[selected_name])])
    production_pipeline.fit(X, y)
    joblib.dump(production_pipeline, ARTIFACT_DIR / "model.joblib")

    source_sha = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    numeric_ranges = {
        "year": {"min": int(cleaned["year"].min()), "max": int(cleaned["year"].max())},
        "km_driven": {"min": int(cleaned["km_driven"].min()), "max": int(cleaned["km_driven"].max())},
    }
    options = {field: sorted(cleaned[field].dropna().unique().tolist()) for field in CATEGORICAL_FEATURES}
    models_by_brand: dict[str, list[str]] = {}
    for model_name in options["name"]:
        models_by_brand.setdefault(extract_brand(model_name), []).append(model_name)
    options["brand"] = sorted(models_by_brand)
    models_by_brand = {brand: sorted(models) for brand, models in sorted(models_by_brand.items())}
    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "dataset": {
            "source_filename": SOURCE_PATH.name,
            "source_sha256": source_sha,
            "rows_before_cleaning": cleaning["rows_before"],
            "rows_after_cleaning": cleaning["rows_after"],
            "duplicate_rows_removed": cleaning["duplicate_rows_removed"],
            "invalid_rows_removed": cleaning["invalid_rows_removed"],
            "missing_values_before": cleaning["missing_values_before"],
            "missing_values_after": cleaning["missing_values_after"],
            "price_min": int(cleaned[TARGET].min()),
            "price_max": int(cleaned[TARGET].max()),
            "price_median": int(cleaned[TARGET].median()),
            "price_mean": float(cleaned[TARGET].mean()),
            "year_min": numeric_ranges["year"]["min"],
            "year_max": numeric_ranges["year"]["max"],
        },
        "numeric_ranges": numeric_ranges,
        "options": options,
        "models_by_brand": models_by_brand,
        "cleaning": cleaning,
    }
    metrics = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "test_rows": len(X_test),
        "selected_model": selected_name,
        "selection_metric": "rmse",
        "selection_rule": "Lowest held-out RMSE; MAE and R² are reported alongside it.",
        "selected_metrics": next(item for item in comparisons if item["model"] == selected_name),
        "cross_validation": cv_metrics,
        "comparisons": comparisons,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "cleaning.json").write_text(json.dumps(cleaning, indent=2), encoding="utf-8")

    print(json.dumps({"selected_model": selected_name, "selected_metrics": metrics["selected_metrics"], "cleaning": cleaning}, indent=2))


if __name__ == "__main__":
    main()
