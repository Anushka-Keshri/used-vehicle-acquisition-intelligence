"""Data loading, schema mapping, cleaning and synthetic fallback data.

Target variable: observed LISTING price (not completed-sale price).

Run standalone:  python -m src.preprocessing
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CURRENT_YEAR: int = datetime.now().year

SCHEMA: list[str] = [
    "brand", "model", "variant", "year", "km_driven",
    "fuel_type", "transmission", "city", "price",
]
REQUIRED_COLUMNS: list[str] = [
    "brand", "model", "year", "km_driven", "fuel_type", "transmission", "price",
]
CATEGORICAL_FEATURES: list[str] = [
    "brand", "model", "variant", "fuel_type", "transmission", "city",
]
NUMERIC_FEATURES: list[str] = ["vehicle_age", "km_driven"]
FEATURES: list[str] = [
    "brand", "model", "variant", "vehicle_age", "km_driven",
    "fuel_type", "transmission", "city",
]
TARGET: str = "price"

COLUMN_ALIASES: dict[str, list[str]] = {
    "brand": ["brand", "make", "manufacturer", "company", "car_brand", "car_make", "brand_name"],
    "model": ["model", "car_model", "model_name"],
    "variant": ["variant", "trim", "version", "variant_name", "car_variant"],
    "year": ["year", "manufacture_year", "manufacturing_year", "mfg_year", "model_year",
             "registration_year", "reg_year", "yr", "yom"],
    "km_driven": ["km_driven", "kms_driven", "kilometers_driven", "kilometres_driven", "km",
                  "kms", "odometer", "distance_driven", "mileage_km", "driven_kms", "kilometer"],
    "fuel_type": ["fuel_type", "fuel", "fueltype"],
    "transmission": ["transmission", "transmission_type", "gearbox", "gear_type", "drive_mode"],
    "city": ["city", "location", "registered_city", "listing_city", "registration_city"],
    "price": ["price", "listing_price", "selling_price", "sale_price", "asking_price",
              "car_price", "price_inr", "price_in_rs", "current_price", "price_lakh",
              "price_in_lakhs"],
}
FULL_NAME_COLUMNS: list[str] = ["name", "car_name", "title", "full_name"]
KNOWN_MULTIWORD_MODELS: list[str] = [
    "Wagon R", "Grand i10 Nios", "Grand i10", "Alto K10", "Alto 800",
    "Innova Crysta", "Innova Hycross", "Santro Xing", "Zen Estilo",
    "Swift Dzire", "i20 Active", "Scorpio Classic", "Scorpio N",
    "Urban Cruiser", "S Cross", "Baleno RS", "XUV 300", "XUV 500", "XUV 700",
    "XUV300", "XUV500", "XUV700",
]
NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "--"}

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_number(value: Any) -> float:
    """Parse messy numeric strings such as '₹6.5 Lakh', '₹8.46L₹5.05 lakh', '65,000 km' or '65k'.

    Handles currency symbols, commas, 'lakh/lac/l', 'crore/cr' and a 'k' suffix.
    For dual prices like '₹8.46L₹5.05 lakh', splits on '₹' and takes the last amount.
    Returns NaN if no number is found.
    """
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    raw_s = str(value).strip()
    if "₹" in raw_s:
        amounts = [p.strip() for p in raw_s.split("₹") if p.strip()]
        s = amounts[-1].lower().replace(",", "") if amounts else ""
    else:
        s = raw_s.lower().replace(",", "")
    for token in ("rs.", "rs", "inr"):
        s = s.replace(token, "")
    s = s.strip()
    match = _NUM_RE.search(s)
    if not match:
        return np.nan
    number = float(match.group())
    if "crore" in s or re.search(r"\bcr\b", s):
        number *= 1e7
    elif "lakh" in s or "lac" in s or re.search(r"\bl\b", s) or re.search(r"\d\s*l(?![a-z])", s):
        number *= 1e5
    elif re.search(r"\d\s*k(?![a-z])", s):
        number *= 1e3
    return number


def _tidy_text(value: Any, brand: bool = False) -> Any:
    """Collapse whitespace, map null tokens to NaN, title-case all-lowercase brands."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return np.nan
    text = " ".join(str(value).split())
    if text.lower() in NULL_TOKENS:
        return np.nan
    if brand and text.islower():
        text = text.title()
    return text


def canon_fuel(value: str) -> str:
    """Map fuel strings to a canonical label (Petrol, Diesel, CNG, LPG, Electric, Hybrid)."""
    s = str(value).lower()
    if "petrol" in s:
        return "Petrol"
    if "diesel" in s:
        return "Diesel"
    if "cng" in s:
        return "CNG"
    if "lpg" in s:
        return "LPG"
    if "electric" in s or s.strip() == "ev":
        return "Electric"
    if "hybrid" in s:
        return "Hybrid"
    return str(value).strip().title()


def canon_transmission(value: str) -> str:
    """Map transmission strings to 'Manual' or 'Automatic' where possible."""
    s = str(value).lower().strip()
    if "auto" in s or s in {"at", "amt", "cvt", "dct", "dsg", "tc"}:
        return "Automatic"
    if "manual" in s or s == "mt":
        return "Manual"
    return str(value).strip().title()


def _normalise_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


# --------------------------------------------------------------------------- #
# Loading and column mapping
# --------------------------------------------------------------------------- #
def load_raw_file(path: Path) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    raise ValueError(f"Unsupported file type: {path.suffix}")


def find_dataset(raw_dir: Path = RAW_DIR) -> Path | None:
    """Return the largest CSV/Parquet file in raw_dir, or None if there is none."""
    if not raw_dir.exists():
        return None
    files = [p for p in raw_dir.iterdir() if p.suffix.lower() in {".csv", ".parquet", ".pq"}]
    return max(files, key=lambda p: p.stat().st_size) if files else None


def map_columns(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename common column-name variants to the canonical schema.

    Falls back to splitting a full-name column ('Maruti Swift VXI') into
    brand/model/variant. Raises ValueError if required columns are missing.
    """
    norm_to_orig: dict[str, Any] = {}
    for col in raw.columns:
        norm_to_orig.setdefault(_normalise_name(col), col)

    mapping: dict[str, str] = {}
    used: set[Any] = set()
    out = pd.DataFrame(index=raw.index)
    for canon in SCHEMA:
        for alias in COLUMN_ALIASES[canon]:
            orig = norm_to_orig.get(alias)
            if orig is not None and orig not in used:
                out[canon] = raw[orig]
                mapping[canon] = str(orig)
                used.add(orig)
                break

    if "brand" not in out.columns and "model" not in out.columns:
        for alias in FULL_NAME_COLUMNS:
            orig = norm_to_orig.get(alias)
            if orig is not None and orig not in used:
                tokens = raw[orig].astype(str).str.split()
                out["brand"] = tokens.map(lambda t: t[0] if len(t) > 0 else np.nan)
                out["model"] = tokens.map(lambda t: t[1] if len(t) > 1 else np.nan)
                out["variant"] = tokens.map(lambda t: " ".join(t[2:]) if len(t) > 2 else np.nan)
                mapping["brand"] = mapping["model"] = f"{orig} (split)"
                break

    if "model" not in out.columns and "variant" in out.columns:
        derived_models = []
        clean_variants = []
        for val in out["variant"]:
            if pd.isna(val) or not str(val).strip():
                derived_models.append(np.nan)
                clean_variants.append(val)
                continue
            s = " ".join(str(val).split())
            matched_model = None
            for km in sorted(KNOWN_MULTIWORD_MODELS, key=len, reverse=True):
                if s.lower().startswith(km.lower()):
                    matched_model = km
                    s_rest = s[len(km):].strip()
                    break
            if matched_model is None:
                parts = s.split(maxsplit=1)
                matched_model = parts[0]
                s_rest = parts[1].strip() if len(parts) > 1 else ""
            derived_models.append(matched_model.title())
            clean_variants.append(s_rest if s_rest else np.nan)
        out["model"] = derived_models
        out["variant"] = clean_variants
        mapping["model"] = f"{mapping.get('variant', 'variant')} (derived)"

    for optional in ("variant", "city"):
        if optional not in out.columns:
            out[optional] = np.nan
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns {missing}. Found: {list(raw.columns)}"
        )
    return out[SCHEMA], mapping


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_dataframe(
    raw: pd.DataFrame, reference_year: int = CURRENT_YEAR
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a raw listings DataFrame and return (clean_df, quality_report).

    Steps: map columns -> parse numbers -> tidy text -> drop duplicates ->
    drop rows missing critical fields -> remove impossible values ->
    remove outliers (3xIQR on log price, 99.5th percentile on km) ->
    engineer vehicle_age.
    """
    report: dict[str, Any] = {"raw_rows": int(len(raw))}
    df, mapping = map_columns(raw)
    report["column_mapping"] = mapping

    price_raw_text = df["price"].astype(str)
    price_has_units = bool(price_raw_text.str.contains(r"lakh|lac|crore|cr|\bl\b|\d\s*l(?![a-z])", case=False, regex=True).any())

    for col in ("price", "year", "km_driven"):
        df[col] = df[col].map(parse_number)
    if not price_has_units and df["price"].notna().any() and df["price"].median() < 1000:
        df["price"] = df["price"] * 1e5
        report["price_scaled_from_lakh"] = True

    for col in ("brand", "model", "variant", "city"):
        df[col] = df[col].map(lambda v, c=col: _tidy_text(v, brand=(c == "brand")))
    for col, fn in (("fuel_type", canon_fuel), ("transmission", canon_transmission)):
        df[col] = df[col].map(_tidy_text).map(lambda v, f=fn: f(v) if isinstance(v, str) else np.nan)

    report["missing_values"] = {k: int(v) for k, v in df.isna().sum().items()}

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(before - len(df))

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    report["rows_dropped_missing_critical"] = int(before - len(df))
    df["variant"] = df["variant"].fillna("Unknown")
    df["city"] = df["city"].fillna("Unknown")

    before = len(df)
    valid = (
        df["price"].between(20_000, 50_000_000)
        & df["year"].between(1995, reference_year)
        & df["km_driven"].between(0, 600_000)
    )
    df = df[valid]
    report["impossible_values_removed"] = int(before - len(df))

    if len(df) < 50:
        raise ValueError("Fewer than 50 valid rows after cleaning; dataset unusable.")

    before = len(df)
    log_price = np.log(df["price"])
    q1, q3 = np.percentile(log_price, [25, 75])
    iqr = q3 - q1
    keep = log_price.between(q1 - 3 * iqr, q3 + 3 * iqr)
    keep &= df["km_driven"] <= np.percentile(df["km_driven"], 99.5)
    df = df[keep]
    report["outliers_removed"] = int(before - len(df))

    df = df.copy()
    df["year"] = df["year"].round().astype(int)
    df["km_driven"] = df["km_driven"].astype(float)
    df["vehicle_age"] = (reference_year - df["year"]).astype(float)
    df = df.reset_index(drop=True)
    report["final_rows"] = int(len(df))
    return df[SCHEMA + ["vehicle_age"]], report


def print_quality_report(report: dict[str, Any], source: str) -> None:
    """Print a human-readable data-quality report."""
    print("=" * 60)
    print(f"DATA QUALITY REPORT  |  source: {source}")
    print("=" * 60)
    print(f"Raw rows                      : {report['raw_rows']:,}")
    print(f"Duplicates removed            : {report['duplicates_removed']:,}")
    print(f"Dropped (missing critical)    : {report['rows_dropped_missing_critical']:,}")
    print(f"Impossible values removed     : {report['impossible_values_removed']:,}")
    print(f"Outliers removed              : {report['outliers_removed']:,}")
    print(f"Final rows                    : {report['final_rows']:,}")
    print("Missing / unparseable values per column (before cleaning):")
    for col, n in report["missing_values"].items():
        print(f"   {col:<14}{n:>8,}")
    print("Column mapping:", report["column_mapping"])
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Synthetic fallback dataset
# --------------------------------------------------------------------------- #
# model -> (approx. new-car price in INR, popularity weight, fuel mix)
_CATALOG: dict[str, dict[str, Any]] = {
    "Maruti Suzuki": {
        "variants": ["LXI", "VXI", "ZXI", "ZXI+"],
        "models": {
            "Swift": (650_000, 10, {"Petrol": 0.85, "CNG": 0.15}),
            "Baleno": (800_000, 8, {"Petrol": 0.9, "CNG": 0.1}),
            "Dzire": (700_000, 8, {"Petrol": 0.8, "CNG": 0.2}),
            "Brezza": (1_000_000, 6, {"Petrol": 0.8, "Diesel": 0.2}),
            "Ertiga": (1_050_000, 6, {"Petrol": 0.7, "Diesel": 0.2, "CNG": 0.1}),
        },
    },
    "Hyundai": {
        "variants": ["E", "S", "SX", "SX(O)"],
        "models": {
            "i20": (850_000, 7, {"Petrol": 0.75, "Diesel": 0.25}),
            "Creta": (1_300_000, 8, {"Petrol": 0.5, "Diesel": 0.5}),
            "Venue": (1_000_000, 5, {"Petrol": 0.7, "Diesel": 0.3}),
            "Verna": (1_150_000, 4, {"Petrol": 0.6, "Diesel": 0.4}),
        },
    },
    "Honda": {
        "variants": ["S", "V", "VX", "ZX"],
        "models": {
            "City": (1_200_000, 5, {"Petrol": 0.7, "Diesel": 0.3}),
            "Amaze": (750_000, 4, {"Petrol": 0.7, "Diesel": 0.3}),
        },
    },
    "Tata": {
        "variants": ["XE", "XM", "XZ", "XZ+"],
        "models": {
            "Nexon": (1_000_000, 6, {"Petrol": 0.5, "Diesel": 0.5}),
            "Altroz": (800_000, 3, {"Petrol": 0.6, "Diesel": 0.4}),
            "Harrier": (1_800_000, 2, {"Diesel": 1.0}),
        },
    },
    "Mahindra": {
        "variants": ["W4", "W6", "W8", "W10"],
        "models": {
            "XUV300": (1_000_000, 3, {"Petrol": 0.4, "Diesel": 0.6}),
            "Scorpio": (1_450_000, 4, {"Diesel": 1.0}),
            "Thar": (1_500_000, 3, {"Petrol": 0.4, "Diesel": 0.6}),
        },
    },
    "Toyota": {
        "variants": ["G", "GX", "VX", "ZX"],
        "models": {
            "Innova Crysta": (2_000_000, 4, {"Diesel": 0.8, "Petrol": 0.2}),
            "Fortuner": (3_400_000, 2, {"Diesel": 0.9, "Petrol": 0.1}),
            "Glanza": (850_000, 2, {"Petrol": 1.0}),
        },
    },
    "Kia": {
        "variants": ["HTE", "HTK", "HTX", "GTX"],
        "models": {
            "Seltos": (1_400_000, 4, {"Petrol": 0.5, "Diesel": 0.5}),
            "Sonet": (1_000_000, 3, {"Petrol": 0.6, "Diesel": 0.4}),
        },
    },
    "Volkswagen": {
        "variants": ["Trendline", "Comfortline", "Highline", "GT"],
        "models": {
            "Polo": (850_000, 2, {"Petrol": 0.7, "Diesel": 0.3}),
            "Vento": (1_100_000, 2, {"Petrol": 0.6, "Diesel": 0.4}),
        },
    },
}
_CITY_MULT: dict[str, tuple[float, float]] = {  # city -> (price multiplier, weight)
    "Bengaluru": (1.02, 14), "Delhi": (0.97, 14), "Mumbai": (1.03, 12),
    "Hyderabad": (1.00, 11), "Chennai": (0.99, 8), "Pune": (1.01, 9),
    "Kolkata": (0.95, 6), "Ahmedabad": (0.97, 6), "Jaipur": (0.94, 5),
    "Lucknow": (0.93, 4),
}
_VARIANT_MULT = [0.90, 1.00, 1.10, 1.20]
_VARIANT_P = [0.20, 0.35, 0.30, 0.15]
_FUEL_MULT = {"Petrol": 1.00, "Diesel": 1.12, "CNG": 1.06}


def generate_synthetic_data(
    n: int = 3000, seed: int = 42, reference_year: int = CURRENT_YEAR
) -> pd.DataFrame:
    """Generate a SYNTHETIC used-car listing dataset with Indian depreciation patterns.

    Depreciation: about 16% in year 1, then about 9% per year. Higher than expected
    mileage lowers price, automatics and diesels carry a premium, cities add a
    +/- few % multiplier and dealers list about 5% above fair value.
    THIS IS NOT REAL MARKET DATA. It only exists so the demo always runs.
    """
    rng = np.random.default_rng(seed)
    flat = [(b, m, spec) for b, bd in _CATALOG.items() for m, spec in bd["models"].items()]
    weights = np.array([spec[1] for _, _, spec in flat], dtype=float)
    weights /= weights.sum()
    cities = list(_CITY_MULT)
    city_w = np.array([_CITY_MULT[c][1] for c in cities], dtype=float)
    city_w /= city_w.sum()
    age_w = np.array([14, 16, 15, 13, 11, 9, 7, 5, 4, 3, 2, 1], dtype=float)
    age_w /= age_w.sum()

    rows = []
    for _ in range(n):
        brand, model, (base, _pop, fuel_mix) = flat[rng.choice(len(flat), p=weights)]
        variants = _CATALOG[brand]["variants"]
        v_idx = rng.choice(4, p=_VARIANT_P)
        fuels = list(fuel_mix)
        fuel = fuels[rng.choice(len(fuels), p=np.array(list(fuel_mix.values())) / sum(fuel_mix.values()))]
        transmission = "Automatic" if rng.random() < 0.30 else "Manual"
        city = cities[rng.choice(len(cities), p=city_w)]
        age = int(rng.choice(np.arange(1, 13), p=age_w))

        km_per_year = rng.lognormal(mean=np.log(11_000), sigma=0.35) * (1.25 if fuel == "Diesel" else 1.0)
        km = float(np.clip(age * km_per_year + rng.normal(0, 2000), 2_000, 280_000))
        km = round(km / 500) * 500

        depreciation = 0.84 * 0.91 ** (age - 1)
        km_factor = float(np.clip(np.exp(-1.5e-6 * (km - 12_000 * age)), 0.80, 1.15))
        price = (
            base * _VARIANT_MULT[v_idx] * _FUEL_MULT[fuel]
            * (1.10 if transmission == "Automatic" else 1.0)
            * depreciation * km_factor * _CITY_MULT[city][0]
            * 1.05 * rng.lognormal(0, 0.06)
        )
        price = max(60_000, round(price / 5_000) * 5_000)
        rows.append((brand, model, variants[v_idx], reference_year - age, km,
                     fuel, transmission, city, float(price)))
    return pd.DataFrame(rows, columns=SCHEMA)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def prepare_dataset(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    save: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load (or synthesise) and clean the dataset.

    Returns (clean_df, info) where info = {is_synthetic, source, report}.
    A real file is used if present and valid; otherwise the synthetic fallback.
    """
    df_clean: pd.DataFrame | None = None
    info: dict[str, Any] = {}
    path = find_dataset(raw_dir)
    if path is not None:
        try:
            raw = load_raw_file(path)
            df_clean, report = clean_dataframe(raw)
            info = {"is_synthetic": False, "source": path.name, "report": report}
        except Exception as exc:  # noqa: BLE001 - we want the demo to always run
            print(f"[preprocessing] Could not use {path.name}: {exc}. Falling back to SYNTHETIC data.")
    if df_clean is None:
        raw = generate_synthetic_data(seed=seed)
        df_clean, report = clean_dataframe(raw)
        info = {"is_synthetic": True, "source": "SYNTHETIC generated data (not real market data)",
                "report": report}
    if verbose:
        print_quality_report(info["report"], info["source"])
    if save:
        processed_dir.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(processed_dir / "vehicles_clean.csv", index=False)
    return df_clean, info


if __name__ == "__main__":
    data, meta = prepare_dataset()
    print(data.head())
