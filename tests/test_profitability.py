"""Unit tests for the profitability engine. Run: python -m unittest discover -s tests -t ."""
import unittest

from src.profitability import (
    ACQUIRE, NEGOTIATE, REJECT, apply_condition_adjustment, classify_decision, evaluate_deal,
    expected_margin, format_inr, max_purchase_price, scenario_sensitivity, select_resale_price,
)


class TestProfitability(unittest.TestCase):
    def test_validation_example(self) -> None:
        """Resale 7,00,000; asking 6,00,000; refurb 35k; other 15k; margin 60k; holding 0."""
        self.assertEqual(expected_margin(700_000, 600_000, 35_000, 15_000, 0, 0), 50_000)
        self.assertEqual(max_purchase_price(700_000, 35_000, 15_000, 0, 0, 60_000), 590_000)
        ev = evaluate_deal(700_000, 600_000, 35_000, 15_000, 0, 0, 60_000, 10.0, False)
        self.assertEqual(ev.decision, NEGOTIATE)
        self.assertEqual(ev.expected_margin, 50_000)
        self.assertEqual(ev.max_purchase_price, 590_000)
        self.assertAlmostEqual(ev.margin_pct, 50_000 / 700_000 * 100)

    def test_holding_cost_is_included(self) -> None:
        self.assertEqual(expected_margin(700_000, 600_000, 35_000, 15_000, 30, 300), 41_000)
        self.assertEqual(max_purchase_price(700_000, 35_000, 15_000, 30, 300, 60_000), 581_000)

    def test_decision_boundaries(self) -> None:
        self.assertEqual(classify_decision(590_000, 590_000), ACQUIRE)
        self.assertEqual(classify_decision(649_000, 590_000, 10), NEGOTIATE)
        self.assertEqual(classify_decision(650_000, 590_000, 10), REJECT)
        self.assertEqual(classify_decision(100_000, -5_000, 10), REJECT)

    def test_risk_aware_uses_lower_bound(self) -> None:
        self.assertEqual(select_resale_price(700_000, 620_000, True), 620_000)
        self.assertEqual(select_resale_price(700_000, 620_000, False), 700_000)

    def test_acquire_and_reject_paths(self) -> None:
        self.assertEqual(evaluate_deal(700_000, 500_000, 35_000, 15_000, 0, 0, 60_000).decision, ACQUIRE)
        ev = evaluate_deal(700_000, 750_000, 35_000, 15_000, 0, 0, 60_000)
        self.assertEqual(ev.decision, REJECT)
        self.assertLess(ev.expected_margin, 0)

    def test_missing_inputs_do_not_crash(self) -> None:
        ev = evaluate_deal(700_000, None, float("nan"), -5, None, None, None)  # type: ignore[arg-type]
        self.assertEqual(ev.decision, ACQUIRE)

    def test_sensitivity_grid(self) -> None:
        df = scenario_sensitivity(700_000, 600_000, 35_000, 15_000, 0, 0)
        self.assertEqual(df.shape, (3, 3))
        self.assertEqual(df.iloc[0, 0], 50_000)
        self.assertEqual(df.iloc[1, 0], 50_000 - 35_000)        # resale -5%
        self.assertAlmostEqual(df.iloc[0, 2], 50_000 - 7_000)   # refurb +20%

    def test_condition_adjustment(self) -> None:
        adj = apply_condition_adjustment(1_000_000, 900_000, 1_100_000, "Fair")
        self.assertAlmostEqual(adj["point"], 930_000)
        self.assertAlmostEqual(adj["adjustment_pct"], -7.0)

    def test_indian_formatting(self) -> None:
        self.assertEqual(format_inr(600_000), "₹6,00,000")
        self.assertEqual(format_inr(12_345_678), "₹1,23,45,678")
        self.assertEqual(format_inr(-50_000), "-₹50,000")
        self.assertEqual(format_inr(999), "₹999")


if __name__ == "__main__":
    unittest.main()
