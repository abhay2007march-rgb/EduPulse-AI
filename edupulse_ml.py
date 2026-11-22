"""Training and inference utilities for EduPulse AI."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "demo_model.joblib"
METRICS_PATH = PROJECT_DIR / "artifacts" / "model_metrics.json"

ID_HINTS = ("student_id", "roll", "registration", "email", "phone", "address", "full_name", "student_name")
DERIVED_HINTS = ("risk_level", "risk_group", "performance_band", "pass_status", "predicted_", "target_")


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    for col in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[col].astype(object)
        values = values.map(lambda value: value.strip() if isinstance(value, str) else value)
        frame[col] = values.replace("", np.nan)
    return frame


def suggest_excluded_columns(columns: list[str], target: str) -> list[str]:
    excluded: list[str] = []
    for col in columns:
        low = col.lower().replace(" ", "_")
        if col != target and any(hint in low for hint in ID_HINTS + DERIVED_HINTS):
            excluded.append(col)
    return excluded


def train_regression_model(
    df: pd.DataFrame,
    target_column: str = "final_score",
    excluded_columns: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a mixed-type Random Forest regression pipeline and return its bundle."""
    frame = _clean_frame(df)
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")
    target = pd.to_numeric(frame[target_column], errors="coerce")
    valid_rows = target.notna()
    frame = frame.loc[valid_rows].copy()
    target = target.loc[valid_rows].astype(float)
    if len(frame) < 30:
        raise ValueError("At least 30 rows with a numeric target are required to train a model.")
    if target.nunique() < 2:
        raise ValueError("The target column must contain at least two different numeric values.")

    excluded = set(excluded_columns or [])
    excluded.add(target_column)
    feature_columns = [c for c in frame.columns if c not in excluded]
    feature_columns = [c for c in feature_columns if frame[c].notna().any() and frame[c].nunique(dropna=True) > 1]
    if not feature_columns:
        raise ValueError("No usable feature columns remain. Keep at least one non-empty input column.")

    X = frame[feature_columns].copy()
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in feature_columns if c not in numeric_features]
    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features))
    if categorical_features:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=350, max_depth=16, min_samples_leaf=2,
            max_features=1.0, random_state=random_state, n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=test_size, random_state=random_state
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, predicted))
    rmse = float(np.sqrt(mean_squared_error(y_test, predicted)))
    r2 = float(r2_score(y_test, predicted)) if len(y_test) > 1 else float("nan")

    regressor = model.named_steps["regressor"]
    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
    raw_importance: dict[str, float] = {c: 0.0 for c in feature_columns}
    for transformed_name, importance in zip(transformed_names, regressor.feature_importances_):
        token = str(transformed_name)
        original = None
        if token.startswith("numeric__"):
            original = token.split("__", 1)[1]
        elif token.startswith("categorical__"):
            encoded = token.split("__", 1)[1]
            matches = [c for c in categorical_features if encoded == c or encoded.startswith(c + "_")]
            if matches:
                original = max(matches, key=len)
        if original in raw_importance:
            raw_importance[original] += float(importance)
    importance_rows = [
        {"feature": name, "importance": value}
        for name, value in sorted(raw_importance.items(), key=lambda item: item[1], reverse=True)
    ]

    feature_metadata: dict[str, dict[str, Any]] = {}
    for col in feature_columns:
        if col in numeric_features:
            values = pd.to_numeric(X[col], errors="coerce").dropna()
            feature_metadata[col] = {
                "kind": "numeric",
                "minimum": float(values.min()) if not values.empty else 0.0,
                "maximum": float(values.max()) if not values.empty else 100.0,
                "median": float(values.median()) if not values.empty else 0.0,
                "integer": bool(values.apply(lambda v: float(v).is_integer()).all()) if not values.empty else False,
            }
        else:
            choices = X[col].dropna().astype(str).value_counts().index.tolist()[:100]
            feature_metadata[col] = {"kind": "categorical", "choices": choices or ["Unknown"],
                                     "default": choices[0] if choices else "Unknown"}

    target_name = target_column.lower()
    if any(word in target_name for word in ("score", "mark", "percent", "grade")) and target.min() >= 0 and target.max() <= 100:
        risk_thresholds = {"high": 40.0, "moderate": 55.0, "method": "fixed_0_100_score_bands"}
    else:
        risk_thresholds = {"high": float(target.quantile(0.25)),
                           "moderate": float(target.quantile(0.50)), "method": "training_target_quartiles"}

    test_results = pd.DataFrame({
        "actual": y_test.to_numpy(dtype=float),
        "predicted": np.asarray(predicted, dtype=float),
        "residual": y_test.to_numpy(dtype=float) - np.asarray(predicted, dtype=float),
    }, index=y_test.index).sort_index()

    bundle: dict[str, Any] = {
        "model": model, "feature_columns": feature_columns,
        "feature_metadata": feature_metadata, "numeric_features": numeric_features,
        "categorical_features": categorical_features, "target_column": target_column,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2,
                    "test_rows": int(len(y_test)), "train_rows": int(len(y_train))},
        "feature_importance": importance_rows, "risk_thresholds": risk_thresholds,
        "target_min": float(target.min()), "target_max": float(target.max()),
        "sample_count": int(len(frame)), "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "Random Forest Regressor", "model_version": "1.0",
    }
    bundle["test_results"] = test_results.reset_index(drop=True)
    return bundle


def risk_label(value: float, thresholds: dict[str, Any]) -> str:
    if float(value) < float(thresholds["high"]):
        return "High risk"
    if float(value) < float(thresholds["moderate"]):
        return "Moderate risk"
    return "Low risk"


def make_predictions(bundle: dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
    """Predict target values; extra columns are ignored and missing features are imputed."""
    feature_columns = bundle["feature_columns"]
    missing = [c for c in feature_columns if c not in data.columns]
    if missing:
        raise ValueError("Uploaded data is missing required feature columns: " + ", ".join(missing))
    X = _clean_frame(data[feature_columns])
    for col in bundle.get("numeric_features", []):
        X[col] = pd.to_numeric(X[col], errors="coerce")
    prediction = bundle["model"].predict(X)
    output = data.copy()
    output[f"predicted_{bundle['target_column']}"] = prediction
    output["predicted_risk"] = [risk_label(v, bundle["risk_thresholds"]) for v in prediction]
    return output


def save_bundle(bundle: dict[str, Any], path: str | Path = MODEL_PATH) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target_path, compress=3)
    return target_path


def load_bundle(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    return joblib.load(Path(path))


def export_metrics(bundle: dict[str, Any], path: str | Path = METRICS_PATH) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in bundle.items() if k not in {"model", "test_results"}}
    target_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target_path
