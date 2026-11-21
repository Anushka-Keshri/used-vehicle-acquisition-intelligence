# Used-Vehicle Acquisition Intelligence

A pre-acquisition decision-support tool for used-car dealerships (hackathon demo, Tekion Theme 2: Data Analytics).

> **We don't just predict prices, we convert predictions into financial decisions.**

It answers three questions before a dealer buys a car:

1. **What is it worth?** An ML market *listing-price* estimate with an 80% range.
2. **Will it be profitable?** Expected margin after refurbishment, other expenses and holding cost.
3. **Should we buy it?** ACQUIRE / NEGOTIATE / REJECT, plus the maximum price we should pay.

It is positioned as a potential pre-acquisition layer, not a claim that Tekion lacks this. It is market-agnostic. In production it would plug into real DMS data (actual sale prices, recon costs).

## Project structure

```
used_vehicle_intelligence/
├── data/raw/            # drop your CSV/Parquet here (optional)
├── data/processed/      # cleaned data written here
├── models/price_model.pkl
├── src/preprocessing.py # load, map columns, clean, synthetic fallback, quality report
├── src/train_model.py   # CatBoost on log(price) (XGBoost fallback), metrics, 80% interval
├── src/predict_price.py # predict(vehicle) -> {point, low, high, shap_values}
├── src/profitability.py # pure finance + decision logic
├── tests/test_profitability.py
├── app.py               # Streamlit UI
└── requirements.txt
```

## Setup (Python 3.10 – 3.12)

```bash
cd used_vehicle_intelligence
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p data/raw data/processed models
python -m src.train_model                              # optional: app auto-trains if missing
python -m unittest discover -s tests -t .              # profitability unit tests
streamlit run app.py
```

## Using real data

Put a CSV or Parquet file in `data/raw/` (e.g., a Cars24 used-car CSV or TCO India `used_listings.parquet`), then run `python -m src.train_model`. Expected schema: `brand, model, variant, year, km_driven, fuel_type, transmission, city, price`. Common column-name variants are mapped automatically. Prices like `₹6.5 Lakh` and kms like `65,000 km` are parsed. If the file is unusable, the app falls back to the synthetic dataset and says so in a banner.

**Without a dataset** the app trains on a clearly labelled **SYNTHETIC** dataset (~3,000 rows) so the demo always runs.

## How decisions are made

```
expected_margin    = resale − asking − refurb − other − holding_days × cost_per_day
max_purchase_price = resale − refurb − other − holding_cost − desired_margin
```

- Risk-aware toggle (sidebar): ON uses the **lower** bound of the 80% range as resale. OFF uses the point estimate.
- **ACQUIRE** if asking ≤ max. **NEGOTIATE** if asking is above max but within the tolerance (default 10%). **REJECT** otherwise.
- Validation example (unit-tested): resale ₹7,00,000, asking ₹6,00,000, refurb ₹35,000, other ₹15,000, desired margin ₹60,000, holding 0 → margin ₹50,000, max price ₹5,90,000, **NEGOTIATE** (target ≤ ₹5,90,000).

## 60-second demo script

1. Click **⚡ Demo scenario** (2019 Hyundai Creta, 65,000 km, Petrol, Manual, Bengaluru, asking ₹6,00,000).
2. Section 02: the market estimate, the 80% range, and how far the asking price sits from market. Open the SHAP tab to show *why* the car is priced this way.
3. Section 03: refurbishment, expenses, holding cost and desired margin turn the estimate into an expected margin.
4. Section 04: read the badge and the maximum price we should pay.
5. Flip **Risk-aware mode** in the sidebar and show the decision tighten when we use the lower bound.

## Limitations

- The target is the observed **listing** price, not the completed-sale price. Sale prices are usually lower.
- Small dataset. Rare brands, variants and cities are less reliable. Unseen categories trigger a warning.
- Refurbishment, other expenses and holding costs are **user-supplied, not predicted**.
- The condition adjustment is a fixed, transparent %, not learned from inspection data.
- The 80% interval is empirical from validation residuals and assumes similar error spread for new cars.
- Synthetic fallback data illustrates the workflow only; it is not real market data.
