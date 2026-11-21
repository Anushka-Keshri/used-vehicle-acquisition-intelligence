"""Train the market listing-price estimator and save model + metadata.

Run:  python -m src.train_model
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    BASE_DIR,
    CATEGORICAL_FEATURES,
    CURRENT_YEAR,
    FEATURES,
    RAW_DIR,
    prepare_dataset,
)

try:  # CatBoost is preferred; XGBoost is the fallback.
    from catboost import CatBoostRegressor, Pool

    CATBOOST_AVAILABLE = True
except Exception:  # noqa: BLE001
    CATBOOST_AVAILABLE = False

MODEL_PATH = BASE_DIR / "models" / "price_model.pkl"


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select model features in fixed order with correct dtypes."""
    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    X["vehicle_age"] = X["vehicle_age"].astype(float)
    X["km_driven"] = X["km_driven"].astype(float)
    return X


def xgb_frame(X: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Convert categorical columns to pandas Categorical with fixed categories.

    Unseen categories become NaN, which XGBoost handles as missing.
    """
    out = X.copy()
    for col in CATEGORICAL_FEATURES:
        out[col] = pd.Categorical(out[col], categories=categories[col])
    return out


def predict_log(model: Any, backend: str, X: pd.DataFrame, categories: dict[str, list[str]]) -> np.ndarray:
    """Predict log(price) with either backend."""
    if backend == "catboost":
        return np.asarray(model.predict(X), dtype=float)
    return np.asarray(model.predict(xgb_frame(X, categories)), dtype=float)


def _fit(
    X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
    categories: dict[str, list[str]], seed: int,
) -> tuple[Any, str]:
    """Fit CatBoost (or XGBoost fallback) on log-price with early stopping."""
    if CATBOOST_AVAILABLE:
        model = CatBoostRegressor(
            iterations=2000, learning_rate=0.05, depth=6, loss_function="RMSE",
            random_seed=seed, verbose=0, early_stopping_rounds=100,
            cat_features=CATEGORICAL_FEATURES, allow_writing_files=False,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        return model, "catboost"
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
        early_stopping_rounds=50, random_state=seed,
    )
    model.fit(xgb_frame(X_tr, categories), y_tr,
              eval_set=[(xgb_frame(X_val, categories), y_val)], verbose=False)
    return model, "xgboost"


def train_and_save(
    raw_dir: Path = RAW_DIR,
    model_path: Path = MODEL_PATH,
    verbose: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full pipeline: data -> train -> evaluate -> calibrate interval -> save.

    Split: 70% train / 15% validation (early stopping + interval calibration) /
    15% test (reported metrics). Returns the saved artifact dictionary.
    """
    df, info = prepare_dataset(raw_dir=raw_dir, verbose=verbose)
    X = build_feature_frame(df)
    y = df["price"].to_numpy(dtype=float)
    categories = {c: sorted(X[c].unique().tolist()) for c in CATEGORICAL_FEATURES}

    idx_tv, idx_test = train_test_split(np.arange(len(df)), test_size=0.15, random_state=seed)
    idx_tr, idx_val = train_test_split(idx_tv, test_size=0.15 / 0.85, random_state=seed)

    model, backend = _fit(X.iloc[idx_tr], np.log(y[idx_tr]), X.iloc[idx_val], np.log(y[idx_val]),
                          categories, seed)

    # 80% prediction interval from empirical validation residual ratios (actual / predicted).
    pred_val = np.exp(predict_log(model, backend, X.iloc[idx_val], categories))
    ratio = y[idx_val] / pred_val
    q_low, q_high = float(np.quantile(ratio, 0.10)), float(np.quantile(ratio, 0.90))

    # Held-out test metrics (INR scale).
    pred_test = np.exp(predict_log(model, backend, X.iloc[idx_test], categories))
    y_test = y[idx_test]
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "r2": float(r2_score(y_test, pred_test)),
        "mape_pct": float(np.mean(np.abs(y_test - pred_test) / y_test) * 100),
        "coverage_80_pct": float(np.mean((y_test >= pred_test * q_low) & (y_test <= pred_test * q_high)) * 100),
        "n_train": int(len(idx_tr)), "n_val": int(len(idx_val)), "n_test": int(len(idx_test)),
    }

    hierarchy: dict[str, dict[str, list[str]]] = {}
    for b, m, v in df[["brand", "model", "variant"]].drop_duplicates().itertuples(index=False):
        hierarchy.setdefault(b, {}).setdefault(m, []).append(v)
    hierarchy = {b: {m: sorted(vs) for m, vs in sorted(ms.items())} for b, ms in sorted(hierarchy.items())}

    defaults: dict[str, Any] = {c: str(df[c].mode().iloc[0]) for c in CATEGORICAL_FEATURES}
    defaults["year"] = int(df["year"].median())
    defaults["km_driven"] = float(df["km_driven"].median())

    artifacts = {
        "model": model,
        "backend": backend,
        "features": FEATURES,
        "categories": categories,
        "hierarchy": hierarchy,
        "options": {c: categories[c] for c in ("fuel_type", "transmission", "city")},
        "defaults": defaults,
        "reference_year": CURRENT_YEAR,
        "q_low": q_low,
        "q_high": q_high,
        "metrics": metrics,
        "is_synthetic": info["is_synthetic"],
        "data_source": info["source"],
        "n_rows": int(len(df)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, model_path)

    if verbose:
        print(f"\nBackend: {backend} | rows: {len(df):,} | saved -> {model_path}")
        print(f"Test MAE  : ₹{metrics['mae']:,.0f}")
        print(f"Test RMSE : ₹{metrics['rmse']:,.0f}")
        print(f"Test R²   : {metrics['r2']:.3f}")
        print(f"Test MAPE : {metrics['mape_pct']:.1f}%")
        print(f"80% interval = estimate x [{q_low:.3f}, {q_high:.3f}] "
              f"(empirical test coverage {metrics['coverage_80_pct']:.1f}%)")
    return artifacts


if __name__ == "__main__":
    train_and_save()
