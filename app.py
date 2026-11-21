"""Used-Vehicle Acquisition Intelligence: Streamlit app.

Run:  streamlit run app.py
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict_price import load_model, predict, shap_to_rupees
from src.profitability import (
    ACQUIRE, CONDITION_ADJUSTMENTS, NEGOTIATE, apply_condition_adjustment, evaluate_deal,
    format_inr, format_lakh, scenario_sensitivity, select_resale_price,
)

st.set_page_config(page_title="Used-Vehicle Acquisition Intelligence", page_icon="🚗", layout="wide")

DECISION_STYLE = {ACQUIRE: ("#1b8a3b", "✅"), NEGOTIATE: ("#e29a00", "🤝"), "REJECT": ("#c62828", "⛔")}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1250px;}
      .badge {color: white; font-weight: 800; font-size: 2.4rem; text-align: center;
              padding: 1.0rem 0.5rem; border-radius: 14px; letter-spacing: 0.08em;}
      .core-msg {font-size: 1.05rem; color: #444; margin-bottom: 0.4rem;}
      .tag {display:inline-block; background:#eef2f7; border-radius:6px; padding:2px 8px;
            font-size:0.78rem; color:#31405a; margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model (first run trains it, ~30-60 seconds)...")
def get_artifacts() -> dict[str, Any]:
    """Load the model artifacts, training the model first if the file is missing."""
    return load_model(auto_train=True)


@st.cache_data(show_spinner=False)
def cached_predict(
    brand: str, model: str, variant: str, year: int, km: float, fuel: str, trans: str, city: str,
    trained_at: str, _artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Cache predictions per vehicle (trained_at invalidates the cache after retraining)."""
    vehicle = {"brand": brand, "model": model, "variant": variant, "year": year, "km_driven": km,
               "fuel_type": fuel, "transmission": trans, "city": city}
    return predict(vehicle, _artifacts)


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #
def _pick(options: list[str], preferred: str, aliases: tuple[str, ...] = ()) -> str:
    """Pick the option matching `preferred` (case-insensitive), else the first option."""
    for name in (preferred, *aliases):
        for opt in options:
            if opt.lower() == name.lower():
                return opt
    return options[0]


def build_demo(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Demo scenario: 2019 Hyundai Creta, 65,000 km, Petrol, Manual, Bengaluru, ask ₹6,00,000."""
    hier, opts = artifacts["hierarchy"], artifacts["options"]
    brand = _pick(sorted(hier), "Hyundai")
    model = _pick(sorted(hier[brand]), "Creta")
    return {
        "brand": brand, "model": model, "variant": _pick(hier[brand][model], "S"),
        "year": min(2019, artifacts["reference_year"]), "km": 65_000,
        "fuel": _pick(opts["fuel_type"], "Petrol"), "trans": _pick(opts["transmission"], "Manual"),
        "city": _pick(opts["city"], "Bengaluru", ("Bangalore",)), "condition": "Good",
        "asking": 600_000, "refurb": 35_000, "other": 15_000,
        "hold_days": 30, "hold_cost": 300, "desired": 60_000,
    }


def apply_demo(demo: dict[str, Any], evaluate: bool = True) -> None:
    """Write the demo scenario into session state (used as an on_click callback)."""
    for key, value in demo.items():
        st.session_state[key] = value
    if evaluate:
        st.session_state["evaluated"] = True


def set_evaluated() -> None:
    """Callback for the Evaluate button."""
    st.session_state["evaluated"] = True


def ensure_option(key: str, options: list[str]) -> None:
    """Reset a selectbox state value if it is not among the current options."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def range_figure(low: float, point: float, high: float, asking: float) -> go.Figure:
    """Horizontal range bar: 80% interval, point estimate and asking price."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[low, high], y=[0, 0], mode="lines+markers+text", name="80% range",
        line=dict(width=18, color="rgba(31,119,180,0.35)"),
        marker=dict(size=10, symbol="line-ns-open", color="#1f77b4"),
        text=[format_inr(low), format_inr(high)], textposition="bottom center", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[point], y=[0], mode="markers+text", name="Market estimate",
        marker=dict(size=22, symbol="diamond", color="#1f77b4"),
        text=[format_inr(point)], textposition="top center",
        hovertemplate="Estimate: " + format_inr(point) + "<extra></extra>"))
    xs = [low, high] + ([asking] if asking > 0 else [])
    pad = (max(xs) - min(xs)) * 0.15 + 1
    if asking > 0:
        fig.add_vline(x=asking, line_dash="dash", line_color="#c62828",
                      annotation_text=f"Asking {format_inr(asking)}", annotation_position="top",
                      annotation_font_color="#c62828")
    fig.update_layout(
        height=270, margin=dict(l=20, r=20, t=50, b=30), showlegend=True,
        legend=dict(orientation="h", y=-0.25),
        xaxis=dict(range=[min(xs) - pad, max(xs) + pad], tickprefix="₹", tickformat=",.0f"),
        yaxis=dict(visible=False, range=[-1, 1]))
    return fig


def shap_figure(expl: dict[str, Any]) -> go.Figure:
    """Horizontal waterfall in rupees: market baseline -> factors -> model estimate."""
    steps = expl["steps"]
    labels = ["Market baseline (avg. listing)"] + [s[0] for s in steps] + ["Model estimate"]
    measures = ["absolute"] + ["relative"] * len(steps) + ["total"]
    values = [expl["base"]] + [s[1] for s in steps] + [expl["final"]]
    texts = [format_inr(expl["base"])] + [
        ("+" if d >= 0 else "−") + format_inr(abs(d)) for _, d in steps] + [format_inr(expl["final"])]
    levels, cum = [expl["base"]], expl["base"]
    for _, d in steps:
        cum += d
        levels.append(cum)
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=measures, y=labels, x=values, text=texts, textposition="outside",
        cliponaxis=False, connector=dict(line=dict(color="#999")),
        increasing=dict(marker=dict(color="#2e9d57")), decreasing=dict(marker=dict(color="#d64545")),
        totals=dict(marker=dict(color="#1f77b4"))))
    fig.update_layout(
        title="Why is this car priced this way?", height=430, margin=dict(l=10, r=60, t=50, b=30),
        xaxis=dict(range=[min(levels) * 0.85, max(levels) * 1.10], tickprefix="₹", tickformat=",.0f"),
        yaxis=dict(autorange="reversed"), showlegend=False)
    return fig


def margin_waterfall(ev: Any, refurb: float, other: float) -> go.Figure:
    """Waterfall: resale -> costs -> expected margin."""
    labels = ["Resale price", "Asking price", "Refurbishment", "Other expenses", "Holding cost", "Expected margin"]
    values = [ev.resale_price, -ev.asking_price, -refurb, -other, -ev.holding_cost, ev.expected_margin]
    texts = [format_inr(v) for v in values]
    fig = go.Figure(go.Waterfall(
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        x=labels, y=values, text=texts, textposition="outside", cliponaxis=False,
        connector=dict(line=dict(color="#999")),
        increasing=dict(marker=dict(color="#2e9d57")), decreasing=dict(marker=dict(color="#d64545")),
        totals=dict(marker=dict(color="#1f77b4"))))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=30),
                      yaxis=dict(tickprefix="₹", tickformat=",.0f"), showlegend=False)
    return fig


def cost_breakdown(ev: Any, refurb: float, other: float) -> tuple[go.Figure, pd.DataFrame]:
    """Donut chart and table of the acquisition cost components."""
    parts = {"Purchase (asking)": ev.asking_price, "Refurbishment": refurb,
             "Other expenses": other, "Holding cost": ev.holding_cost}
    parts = {k: v for k, v in parts.items() if v > 0}
    total = sum(parts.values()) or 1.0
    fig = go.Figure(go.Pie(labels=list(parts), values=list(parts.values()), hole=0.55,
                           textinfo="label+percent"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      annotations=[dict(text=format_lakh(total), x=0.5, y=0.5, showarrow=False,
                                        font=dict(size=18))])
    table = pd.DataFrame({"Component": list(parts), "Amount": [format_inr(v) for v in parts.values()],
                          "Share": [f"{v / total * 100:.1f}%" for v in parts.values()]})
    table.loc[len(table)] = ["Total investment", format_inr(total), "100%"]
    return fig, table


def sensitivity_heatmap(df: pd.DataFrame) -> go.Figure:
    """Annotated margin heatmap (red = loss, green = profit)."""
    fig = go.Figure(go.Heatmap(
        z=df.values, x=list(df.columns), y=list(df.index), colorscale="RdYlGn", zmid=0,
        text=[[format_inr(v) for v in row] for row in df.values], texttemplate="%{text}",
        textfont=dict(size=14), showscale=False,
        hovertemplate="%{y} | %{x}<br>Margin: %{text}<extra></extra>"))
    fig.update_layout(height=290, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    return fig


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
artifacts = get_artifacts()
metrics = artifacts["metrics"]
demo = build_demo(artifacts)

if "initialized" not in st.session_state:
    apply_demo(demo, evaluate=False)
    st.session_state["evaluated"] = False
    st.session_state["initialized"] = True

# ------------------------------- sidebar ------------------------------------ #
with st.sidebar:
    st.header("Model performance")
    st.caption(f"Market listing-price estimator ({artifacts['backend'].title()}), held-out test set")
    c1, c2 = st.columns(2)
    c1.metric("MAE", format_inr(metrics["mae"]))
    c2.metric("RMSE", format_inr(metrics["rmse"]))
    c3, c4 = st.columns(2)
    c3.metric("R²", f"{metrics['r2']:.3f}")
    c4.metric("MAPE", f"{metrics['mape_pct']:.1f}%")
    st.caption(f"80% range = estimate × [{artifacts['q_low']:.2f}, {artifacts['q_high']:.2f}] from validation "
               f"residuals; empirical test coverage {metrics['coverage_80_pct']:.0f}%.")

    st.subheader("Data source")
    st.caption(f"{artifacts['data_source']}  \n{artifacts['n_rows']:,} cleaned rows · "
               f"train {metrics['n_train']:,} / val {metrics['n_val']:,} / test {metrics['n_test']:,}")

    st.subheader("Decision settings")
    risk_aware = st.toggle("Risk-aware (conservative) mode", value=False, key="risk_aware",
                           help="ON: resale = LOWER bound of the 80% range. OFF: resale = point estimate.")
    tolerance = st.slider("Negotiation tolerance (% above max price)", 0, 30, 10, key="tolerance",
                          help="NEGOTIATE if asking is above the maximum price by no more than this.")
    st.caption("ACQUIRE: asking ≤ max price · NEGOTIATE: within tolerance above · REJECT: otherwise")

    with st.expander("⚠️ Limitations"):
        st.markdown(
            "- **Listing price ≠ sale price.** The model learns what cars are *listed* at; "
            "actual sale prices are typically lower after negotiation.\n"
            "- **Small dataset.** Rare brands/variants/cities get less reliable estimates.\n"
            "- **Costs are user-supplied, not predicted.** Refurbishment, expenses and holding costs "
            "come from the dealer.\n"
            "- Condition is a fixed % adjustment, not learned from inspection data.")

# -------------------------------- header ------------------------------------ #
st.title("🚗 Used-Vehicle Acquisition Intelligence")
st.markdown('<div class="core-msg"><b>We don\'t just predict prices, we convert predictions into '
            'financial decisions.</b> A potential pre-acquisition layer for dealer workflows: '
            'what is it worth, will it be profitable, should we buy it?</div>', unsafe_allow_html=True)
st.markdown('<span class="tag">Market-agnostic</span><span class="tag">Would plug into real DMS data: '
            'sale prices &amp; recon costs</span><span class="tag">Market listing-price estimator</span>',
            unsafe_allow_html=True)
if artifacts["is_synthetic"]:
    st.warning("⚠️ **SYNTHETIC DEMO DATA IN USE.** No dataset was found in `data/raw/`, so the model was "
               "trained on generated data with realistic depreciation patterns. Numbers illustrate the "
               "workflow only; they are not real market prices.")

# ------------------------------ 01 input ------------------------------------ #
st.markdown("### 01 · Vehicle input")
hier, opts = artifacts["hierarchy"], artifacts["options"]
cur_year = artifacts["reference_year"]

r1 = st.columns(4)
brands = sorted(hier)
ensure_option("brand", brands)
brand = r1[0].selectbox("Brand", brands, key="brand")
models = sorted(hier[brand])
ensure_option("model", models)
model_name = r1[1].selectbox("Model", models, key="model")
variants = hier[brand][model_name]
ensure_option("variant", variants)
variant = r1[2].selectbox("Variant", variants, key="variant")
st.session_state["year"] = int(min(max(st.session_state.get("year", 2019), 1995), cur_year))
year = r1[3].number_input("Year of manufacture", min_value=1995, max_value=cur_year, step=1, key="year")

r2 = st.columns(4)
km = r2[0].number_input("Km driven", min_value=0, max_value=500_000, step=1000, key="km")
ensure_option("fuel", opts["fuel_type"])
fuel = r2[1].selectbox("Fuel type", opts["fuel_type"], key="fuel")
ensure_option("trans", opts["transmission"])
trans = r2[2].selectbox("Transmission", opts["transmission"], key="trans")
ensure_option("city", opts["city"])
city = r2[3].selectbox("City", opts["city"], key="city")

r3 = st.columns([3, 1, 1])
condition = r3[0].radio(
    "Vehicle condition (transparent price adjustment applied to the model estimate)",
    list(CONDITION_ADJUSTMENTS), horizontal=True, key="condition",
    format_func=lambda c: f"{c} ({CONDITION_ADJUSTMENTS[c] * 100:+.0f}%)")
r3[1].write("")
r3[1].button("🔍 Evaluate Vehicle", type="primary", on_click=set_evaluated, use_container_width=True)
r3[2].write("")
r3[2].button("⚡ Demo scenario", on_click=apply_demo, args=(demo,), use_container_width=True,
             help="2019 Hyundai Creta · 65,000 km · Petrol · Manual · Bengaluru · asking ₹6,00,000")

if not st.session_state.get("evaluated"):
    st.info("Enter the vehicle details (or load the demo scenario) and click **Evaluate Vehicle**.")
    st.stop()

# ----------------------------- 02 valuation --------------------------------- #
st.markdown("---")
st.markdown("### 02 · Market valuation")
res = cached_predict(brand, model_name, variant, int(year), float(km), fuel, trans, city,
                     artifacts["trained_at"], artifacts)
for w in res["warnings"]:
    st.warning(w)
adj = apply_condition_adjustment(res["point"], res["low"], res["high"], condition)
asking_now = int(st.session_state.get("asking", 0) or 0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Market listing-price estimate", format_inr(adj["point"]))
m2.metric("80% price range", f"{format_lakh(adj['low'])} – {format_lakh(adj['high'])}")
if asking_now > 0:
    pct = (asking_now / adj["point"] - 1) * 100
    m3.metric("Asking vs market", f"{pct:+.1f}%", delta="above market" if pct > 0 else "below market",
              delta_color="inverse")
else:
    m3.metric("Asking vs market", "n/a", delta="enter asking price below", delta_color="off")
m4.metric("Condition adjustment", f"{adj['adjustment_pct']:+.0f}% ({condition})",
          delta=f"{format_inr(adj['point'] - res['point'])} vs model", delta_color="off")
st.caption(f"Model output {format_inr(res['point'])} × (1 {adj['adjustment_pct'] / 100:+.2f}) = "
           f"{format_inr(adj['point'])}. The model estimates *listing* prices, not completed-sale prices.")

tab_range, tab_shap = st.tabs(["📏 Price range vs asking", "🔎 Why is this car priced this way?"])
with tab_range:
    st.plotly_chart(range_figure(adj["low"], adj["point"], adj["high"], asking_now),
                    use_container_width=True, key="range_chart")
with tab_shap:
    expl = shap_to_rupees(res["shap_values"], res["base_value"], res["feature_values"])
    st.plotly_chart(shap_figure(expl), use_container_width=True, key="shap_chart")
    st.caption("SHAP attribution of the *model* estimate (before the condition adjustment), "
               "converted to approximate rupee effects, ordered by impact.")

# ----------------------------- 03 financials -------------------------------- #
st.markdown("---")
st.markdown("### 03 · Financial analysis")
f1 = st.columns(3)
asking = f1[0].number_input("Asking price (₹)", min_value=0, step=10_000, key="asking")
f1[0].caption(format_inr(asking))
refurb = f1[1].number_input("Refurbishment cost (₹)", min_value=0, step=5_000, key="refurb")
f1[1].caption(format_inr(refurb))
other = f1[2].number_input("Other expenses (₹): transfer, insurance, docs", min_value=0, step=5_000, key="other")
f1[2].caption(format_inr(other))
f2 = st.columns(3)
hold_days = f2[0].number_input("Expected holding period (days)", min_value=0, max_value=365, step=5, key="hold_days")
hold_cost = f2[1].number_input("Holding cost per day (₹)", min_value=0, step=50, key="hold_cost")
f2[1].caption(f"Total holding cost: {format_inr(hold_days * hold_cost)}")
desired = f2[2].number_input("Desired margin (₹)", min_value=0, step=5_000, key="desired")
f2[2].caption(format_inr(desired))

if asking <= 0:
    st.warning("Asking price is ₹0; enter the seller's price for a meaningful decision.")

resale = select_resale_price(adj["point"], adj["low"], risk_aware)
ev = evaluate_deal(resale, asking, refurb, other, hold_days, hold_cost, desired, tolerance, risk_aware)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Resale price used", format_inr(ev.resale_price),
          delta="lower bound (conservative)" if risk_aware else "point estimate", delta_color="off")
k2.metric("Total investment", format_inr(ev.total_cost))
k3.metric("Expected margin", format_inr(ev.expected_margin),
          delta=f"{format_inr(ev.expected_margin - ev.desired_margin)} vs target")
k4.metric("Margin % of resale", f"{ev.margin_pct:.1f}%")

tab_wf, tab_cost = st.tabs(["💧 Margin waterfall", "🧾 Cost breakdown"])
with tab_wf:
    st.plotly_chart(margin_waterfall(ev, refurb, other), use_container_width=True, key="wf_chart")
with tab_cost:
    pie, table = cost_breakdown(ev, refurb, other)
    cc1, cc2 = st.columns([1, 1])
    cc1.plotly_chart(pie, use_container_width=True, key="pie_chart")
    cc2.dataframe(table, hide_index=True, use_container_width=True)

# --------------------------- 04 recommendation ------------------------------ #
st.markdown("---")
st.markdown("### 04 · Final recommendation")
color, icon = DECISION_STYLE[ev.decision]
left, right = st.columns([1.2, 1])
with left:
    st.markdown(f'<div class="badge" style="background:{color}">{icon} {ev.decision}</div>',
                unsafe_allow_html=True)
    st.write("")
    for line in ev.explanation.split("\n"):
        st.markdown(f"- {line}")
with right:
    st.metric("Maximum acceptable purchase price", format_inr(ev.max_purchase_price))
    a, b = st.columns(2)
    a.metric("Asking price", format_inr(ev.asking_price))
    b.metric("Gap to max", format_inr(ev.gap),
             delta=None if ev.gap_pct is None else f"{ev.gap_pct:+.1f}%", delta_color="inverse")
    if ev.decision == NEGOTIATE:
        st.info(f"🎯 Negotiation target: ≤ {format_inr(ev.max_purchase_price)} · "
                f"walk-away above {format_inr(ev.walk_away_price)}")

st.markdown("#### Scenario sensitivity: margin at the entered asking price")
sens = scenario_sensitivity(ev.resale_price, asking, refurb, other, hold_days, hold_cost)
tab_h, tab_t = st.tabs(["🌡️ Heatmap", "📋 Table"])
with tab_h:
    st.plotly_chart(sensitivity_heatmap(sens), use_container_width=True, key="sens_chart")
with tab_t:
    st.dataframe(sens.apply(lambda col: col.map(format_inr)), use_container_width=True)
st.caption("Rows: resale price drops 0/5/10%. Columns: refurbishment cost rises 0/10/20%. "
           "Red = expected loss.")

st.markdown("---")
st.caption("Decision-support prototype. Estimates are model-based listing-price estimates; "
           "validate against real DMS sale prices and recon costs before relying on them.")
