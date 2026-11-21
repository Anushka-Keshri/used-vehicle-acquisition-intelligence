"""Pure profitability and decision logic (no I/O, no UI, fully unit-testable).

expected_margin    = resale - asking - refurb - other - holding_days * cost_per_day
max_purchase_price = resale - refurb - other - holding_cost - desired_margin
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

ACQUIRE, NEGOTIATE, REJECT = "ACQUIRE", "NEGOTIATE", "REJECT"

# Transparent condition adjustments applied to the model's market estimate.
CONDITION_ADJUSTMENTS: dict[str, float] = {
    "Excellent": 0.03, "Good": 0.0, "Fair": -0.07, "Poor": -0.15,
}


# ----------------------------- formatting ---------------------------------- #
def format_inr(value: float | None, prefix: str = "₹") -> str:
    """Format a number in Indian digit grouping, e.g. 600000 -> '₹6,00,000'."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    n = int(round(value))
    sign = "-" if n < 0 else ""
    digits = str(abs(n))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    return f"{sign}{prefix}{digits}"


def format_lakh(value: float) -> str:
    """Compact Indian format: ₹6.2 L or ₹1.25 Cr."""
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e7:
        return f"{sign}₹{v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"{sign}₹{v / 1e5:.2f} L"
    return f"{sign}₹{v:,.0f}"


# ----------------------------- core maths ---------------------------------- #
def _nn(value: Any) -> float:
    """Coerce to a non-negative float; None/NaN/invalid -> 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return max(v, 0.0)


def holding_cost(holding_days: float, holding_cost_per_day: float) -> float:
    """Total holding cost = days x daily cost."""
    return _nn(holding_days) * _nn(holding_cost_per_day)


def expected_margin(
    resale_price: float, asking_price: float, refurbishment: float, other_expenses: float,
    holding_days: float = 0, holding_cost_per_day: float = 0,
) -> float:
    """Expected profit if bought at asking_price and resold at resale_price."""
    return (_nn(resale_price) - _nn(asking_price) - _nn(refurbishment) - _nn(other_expenses)
            - holding_cost(holding_days, holding_cost_per_day))


def max_purchase_price(
    resale_price: float, refurbishment: float, other_expenses: float,
    holding_days: float = 0, holding_cost_per_day: float = 0, desired_margin: float = 0,
) -> float:
    """Highest price payable that still leaves the desired margin (can be negative)."""
    return (_nn(resale_price) - _nn(refurbishment) - _nn(other_expenses)
            - holding_cost(holding_days, holding_cost_per_day) - _nn(desired_margin))


def select_resale_price(point: float, low: float | None, risk_aware: bool) -> float:
    """Risk-aware mode uses the LOWER bound of the range; otherwise the point estimate."""
    if risk_aware and low is not None and not (isinstance(low, float) and math.isnan(low)):
        return float(low)
    return float(point)


def classify_decision(asking_price: float, max_price: float, tolerance_pct: float = 10.0) -> str:
    """ACQUIRE if asking <= max; NEGOTIATE if within tolerance above max; else REJECT."""
    asking, eps = _nn(asking_price), 1e-9
    if asking <= max_price + eps:
        return ACQUIRE
    if max_price > 0 and asking <= max_price * (1 + _nn(tolerance_pct) / 100) + eps:
        return NEGOTIATE
    return REJECT


def apply_condition_adjustment(point: float, low: float, high: float, condition: str) -> dict[str, float]:
    """Apply the transparent condition adjustment (%) to the model's price and range."""
    adj = CONDITION_ADJUSTMENTS.get(condition, 0.0)
    factor = 1 + adj
    return {"adjustment_pct": adj * 100, "factor": factor,
            "point": point * factor, "low": low * factor, "high": high * factor}


@dataclass(frozen=True)
class DealEvaluation:
    """Result of evaluating a potential acquisition."""

    decision: str
    resale_price: float
    asking_price: float
    expected_margin: float
    margin_pct: float            # expected margin as % of resale price
    max_purchase_price: float
    gap: float                   # asking - max (positive = asking above ceiling)
    gap_pct: float | None        # gap as % of max price (None if max <= 0)
    holding_cost: float
    total_cost: float            # asking + refurb + other + holding
    desired_margin: float
    tolerance_pct: float
    walk_away_price: float       # max price x (1 + tolerance)
    risk_aware: bool
    explanation: str             # lines separated by newlines

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluation as a plain dict."""
        return asdict(self)


def evaluate_deal(
    resale_price: float, asking_price: float, refurbishment: float, other_expenses: float,
    holding_days: float, holding_cost_per_day: float, desired_margin: float,
    negotiation_tolerance_pct: float = 10.0, risk_aware: bool = False,
) -> DealEvaluation:
    """Compute margin, maximum price, decision and a plain-English explanation.

    `resale_price` is the basis already chosen via select_resale_price();
    `risk_aware` is only recorded for the explanation text.
    """
    resale, asking = _nn(resale_price), _nn(asking_price)
    refurb, other, desired = _nn(refurbishment), _nn(other_expenses), _nn(desired_margin)
    hold = holding_cost(holding_days, holding_cost_per_day)
    tol = _nn(negotiation_tolerance_pct)

    margin = expected_margin(resale, asking, refurb, other, holding_days, holding_cost_per_day)
    max_price = max_purchase_price(resale, refurb, other, holding_days, holding_cost_per_day, desired)
    decision = classify_decision(asking, max_price, tol)
    gap = asking - max_price
    gap_pct = (gap / max_price * 100) if max_price > 0 else None
    margin_pct = (margin / resale * 100) if resale > 0 else 0.0
    walk_away = max_price * (1 + tol / 100) if max_price > 0 else 0.0

    explanation = _explain(decision, resale, asking, margin, margin_pct, max_price, gap, gap_pct,
                           desired, tol, walk_away, risk_aware)
    return DealEvaluation(
        decision=decision, resale_price=resale, asking_price=asking, expected_margin=margin,
        margin_pct=margin_pct, max_purchase_price=max_price, gap=gap, gap_pct=gap_pct,
        holding_cost=hold, total_cost=asking + refurb + other + hold, desired_margin=desired,
        tolerance_pct=tol, walk_away_price=walk_away, risk_aware=risk_aware, explanation=explanation,
    )


def _explain(
    decision: str, resale: float, asking: float, margin: float, margin_pct: float, max_price: float,
    gap: float, gap_pct: float | None, desired: float, tol: float, walk_away: float, risk_aware: bool,
) -> str:
    """Build a 3-4 line plain-English explanation."""
    inr = format_inr
    basis = "conservative (lower-bound)" if risk_aware else "point-estimate"
    lines: list[str] = []
    if max_price <= 0:
        lines.append(f"At a {basis} resale value of {inr(resale)}, costs plus the {inr(desired)} target "
                     "margin already exceed expected resale revenue, so no positive purchase price works.")
    else:
        lines.append(f"At a {basis} resale value of {inr(resale)}, the most we can pay and still earn "
                     f"the {inr(desired)} target margin is {inr(max_price)}.")
    if decision == ACQUIRE:
        lines.append(f"The asking price of {inr(asking)} is {inr(-gap)} below that ceiling.")
        lines.append(f"Expected margin at asking is {inr(margin)} ({margin_pct:.1f}% of resale), "
                     f"{inr(margin - desired)} above target.")
        lines.append("Proceed with the purchase; opening a little lower would widen the safety buffer.")
    elif decision == NEGOTIATE:
        lines.append(f"The asking price of {inr(asking)} is {inr(gap)} ({gap_pct:.1f}%) above that ceiling, "
                     f"inside the {tol:.0f}% negotiation tolerance.")
        lines.append(f"At asking, expected margin is {inr(margin)} ({margin_pct:.1f}%), "
                     f"{inr(desired - margin)} short of target.")
        lines.append(f"Negotiate to {inr(max_price)} or lower; walk away above {inr(walk_away)}.")
    else:
        gap_txt = f"{inr(gap)} ({gap_pct:.1f}%)" if gap_pct is not None else inr(gap)
        lines.append(f"The asking price of {inr(asking)} is {gap_txt} above that ceiling, "
                     f"outside the {tol:.0f}% negotiation tolerance.")
        outcome = f"an expected loss of {inr(-margin)}" if margin < 0 else f"an expected margin of {inr(margin)}"
        lines.append(f"At asking, the result would be {outcome} versus a {inr(desired)} target.")
        if max_price > 0:
            lines.append(f"Pass unless the seller drops to about {inr(walk_away)} or less "
                         "(the edge of the negotiation zone), or recon costs prove lower.")
        else:
            lines.append("Pass on this vehicle unless the resale outlook or recon costs change materially.")
    return "\n".join(lines)


def scenario_sensitivity(
    resale_price: float, asking_price: float, refurbishment: float, other_expenses: float,
    holding_days: float, holding_cost_per_day: float,
    resale_drops_pct: tuple[float, ...] = (0, 5, 10), refurb_rises_pct: tuple[float, ...] = (0, 10, 20),
) -> pd.DataFrame:
    """Expected margin under resale-price drops (rows) and refurb-cost rises (columns)."""
    def row_label(d: float) -> str:
        return "Resale as estimated" if d == 0 else f"Resale −{d:g}%"

    def col_label(r: float) -> str:
        return "Refurb as entered" if r == 0 else f"Refurb +{r:g}%"

    data = {
        col_label(r): [
            expected_margin(_nn(resale_price) * (1 - d / 100), asking_price,
                            _nn(refurbishment) * (1 + r / 100), other_expenses,
                            holding_days, holding_cost_per_day)
            for d in resale_drops_pct
        ]
        for r in refurb_rises_pct
    }
    return pd.DataFrame(data, index=[row_label(d) for d in resale_drops_pct])
