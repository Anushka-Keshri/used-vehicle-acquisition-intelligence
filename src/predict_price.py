"""Inference: market listing-price estimate, 80% range and SHAP explanations."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, canon_fuel, canon_transmission
from src.train_model import MODEL_PATH, build_feature_frame, predict_log, train_and_save, xgb_frame

try:
    from catboost import Pool
except Exception:  # noqa: BLE001
    Pool = None  # type: ignore[assignment]

FEATURE_LABELS = {
    "brand": "Brand", "model": "Model", "variant": "Variant", "vehicle_age": "Vehicle age",
    "km_driven": "Km driven", "fuel_type": "Fuel", "transmission": "Transmission", "city": "City",
}
_EXPLAINER_CACHE: dict[int, Any] = {}


def load_model(model_path: Path = MODEL_PATH, auto_train: bool = True) -> dict[str, Any]:
    """Load model artifacts; train first if the file is missing or unreadable."""
    path = Path(model_path)
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            if not auto_train:
                raise
            print(f"[predict_price] Could not load {path.name} ({exc}); retraining.")
    elif not auto_train:
        raise FileNotFoundError(f"Model not found at {path}. Run: python -m src.train_model")
    train_and_save(model_path=path, verbose=False)
    return joblib.load(path)


def _to_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def normalize_vehicle(vehicle: dict[str, Any], artifacts: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fill missing inputs, canonicalise categories and flag unseen values.

    Never raises on bad input; returns (clean_vehicle, warnings).
    """
    warnings: list[str] = []
    defaults, ref_year = artifacts["defaults"], artifacts["reference_year"]
    clean: dict[str, Any] = {}

    for col in CATEGORICAL_FEATURES:
        raw = vehicle.get(col)
        text = "" if raw is None else " ".join(str(raw).split())
        if text.lower() in {"", "nan", "none", "null"}:
            text = defaults[col]
            warnings.append(f"{FEATURE_LABELS[col]} missing; assumed '{text}'.")
        if col == "fuel_type":
            text = canon_fuel(text)
        elif col == "transmission":
            text = canon_transmission(text)
        lookup = {k.lower(): k for k in artifacts["categories"][col]}
        if text.lower() in lookup:
            text = lookup[text.lower()]
        else:
            warnings.append(f"{FEATURE_LABELS[col]} '{text}' was not seen in training data; "
                            "the estimate is less reliable.")
        clean[col] = text

    hierarchy = artifacts["hierarchy"]
    if clean["brand"] in hierarchy and clean["model"] not in hierarchy[clean["brand"]] \
            and clean["model"] in artifacts["categories"]["model"]:
        warnings.append(f"The combination {clean['brand']} {clean['model']} was not seen in training data.")

    year = _to_float(vehicle.get("year"))
    if year is None:
        year = float(defaults["year"])
        warnings.append(f"Year missing; assumed {int(year)}.")
    if year > ref_year or year < 1995:
        year = float(min(max(year, 1995), ref_year))
        warnings.append(f"Year out of range; clipped to {int(year)}.")

    km = _to_float(vehicle.get("km_driven"))
    if km is None:
        km = float(defaults["km_driven"])
        warnings.append(f"Kilometres missing; assumed {km:,.0f} km.")
    if km < 0 or km > 600_000:
        km = float(min(max(km, 0), 600_000))
        warnings.append(f"Kilometres out of range; clipped to {km:,.0f} km.")

    clean["year"] = int(year)
    clean["km_driven"] = float(km)
    clean["vehicle_age"] = float(max(ref_year - int(year), 0))
    return clean, warnings


def _shap_library(artifacts: dict[str, Any], X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP values via the `shap` library (TreeExplainer), in log-price space."""
    import shap

    model, backend = artifacts["model"], artifacts["backend"]
    explainer = _EXPLAINER_CACHE.get(id(model))
    if explainer is None:
        explainer = shap.TreeExplainer(model)
        _EXPLAINER_CACHE[id(model)] = explainer
    data = Pool(X, cat_features=CATEGORICAL_FEATURES) if backend == "catboost" \
        else xgb_frame(X, artifacts["categories"])
    values = np.asarray(explainer.shap_values(data))
    if values.ndim != 2 or values.shape[1] != len(FEATURES):
        raise ValueError("Unexpected SHAP output shape")
    return values[0], float(np.ravel(explainer.expected_value)[0])


def _shap_native(artifacts: dict[str, Any], X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Native TreeSHAP from CatBoost/XGBoost, used if the shap library path fails."""
    model = artifacts["model"]
    if artifacts["backend"] == "catboost":
        vals = model.get_feature_importance(Pool(X, cat_features=CATEGORICAL_FEATURES), type="ShapValues")
    else:
        import xgboost as xgb

        dm = xgb.DMatrix(xgb_frame(X, artifacts["categories"]), enable_categorical=True)
        vals = model.get_booster().predict(dm, pred_contribs=True)
    return np.asarray(vals)[0, :-1], float(np.asarray(vals)[0, -1])


def compute_shap(artifacts: dict[str, Any], X: pd.DataFrame, log_pred: float) -> tuple[dict[str, float], float]:
    """Return ({feature: log-space contribution}, base_value). Never raises."""
    for fn in (_shap_library, _shap_native):
        try:
            values, base = fn(artifacts, X)
            if abs(float(values.sum()) + base - log_pred) < 0.02:
                return {f: float(v) for f, v in zip(FEATURES, values)}, base
        except Exception:  # noqa: BLE001
            continue
    return {f: 0.0 for f in FEATURES}, float(log_pred)


def predict(vehicle: dict[str, Any], artifacts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Estimate market listing price for one vehicle.

    Args:
        vehicle: dict with brand, model, variant, year, km_driven, fuel_type,
            transmission, city. Missing/unseen values are handled with warnings.
        artifacts: loaded artifacts (loaded/trained automatically if None).

    Returns:
        {point, low, high, shap_values, base_value, feature_values, warnings}
        point/low/high are in INR; shap_values are log-price contributions.
    """
    artifacts = artifacts or load_model()
    clean, warnings = normalize_vehicle(vehicle, artifacts)
    row = pd.DataFrame([clean])
    X = build_feature_frame(row)
    log_pred = float(predict_log(artifacts["model"], artifacts["backend"], X, artifacts["categories"])[0])
    point = math.exp(log_pred)
    low, high = sorted((point * artifacts["q_low"], point * artifacts["q_high"]))
    shap_values, base = compute_shap(artifacts, X, log_pred)
    return {
        "point": float(point), "low": float(low), "high": float(high),
        "shap_values": shap_values, "base_value": float(base),
        "feature_values": clean, "warnings": warnings,
    }


def feature_label(feature: str, value: Any) -> str:
    """Human-readable label such as 'Vehicle age: 7 yrs'."""
    if feature == "vehicle_age":
        return f"Vehicle age: {int(value)} yrs"
    if feature == "km_driven":
        return f"Km driven: {int(value):,}"
    return f"{FEATURE_LABELS[feature]}: {value}"


def shap_to_rupees(
    shap_values: dict[str, float], base_value: float, feature_values: dict[str, Any], top_n: int = 6,
) -> dict[str, Any]:
    """Convert log-space SHAP values to an approximate rupee waterfall.

    Features are applied in order of |impact|; each step's rupee effect is the
    change in exp(base + cumulative SHAP). The sum of steps equals
    model estimate minus market baseline exactly; the split between features
    depends on ordering, so treat it as a guide.
    """
    ordered = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    head, tail = ordered[:top_n], ordered[top_n:]
    steps: list[tuple[str, float]] = []
    cum, prev = base_value, math.exp(base_value)
    for feat, val in head:
        cum += val
        cur = math.exp(cum)
        steps.append((feature_label(feat, feature_values[feat]), cur - prev))
        prev = cur
    if tail:
        cum += sum(v for _, v in tail)
        cur = math.exp(cum)
        steps.append(("All other factors", cur - prev))
        prev = cur
    return {"base": math.exp(base_value), "steps": steps, "final": prev}
