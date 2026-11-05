# -*- coding: utf-8 -*-
"""
Self-contained mathematical and point-in-time hygiene unit tests for the
Quantamental Cross-Curve Commodity Engine.

Runs standalone with zero external CSV dependencies:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from quantamental import (
    CASH_RESERVE_SHARE_PCT,
    DAY1_DEPLOYED_SHARE_PCT,
    MANDATE_SLEEVE_SHARES_PCT,
    MANDATE_TARGET_VOL_PCT,
    black76_option,
    compute_contract_matched_returns,
    compute_cot_tradeable_availability,
    compute_monthly_sst_availability,
    evaluate_two_step_option_filter,
    newey_west_tstat,
    select_purged_training_rows,
)


class TestEngineIntegrity(unittest.TestCase):
    """Verify point-in-time lag rules, roll-free returns, purging, Black-76, and mandate invariants."""

    def test_contract_matched_returns_eliminate_contango_roll_jump(self) -> None:
        """
        Construct a synthetic contango curve where each individual YYYY-MM contract's
        price is 100% unchanged week-over-week, but tenor slots roll forward by 1 month
        on week 2. Contract-matched M3 return must be 0.0, while naive continuation M3
        picks up a spurious positive roll jump.
        """
        contract_prices = {
            "2027-01": 100.0,
            "2027-02": 102.0,
            "2027-03": 104.0,
            "2027-04": 106.0,
            "2027-05": 108.0,
            "2027-06": 110.0,
        }
        w1 = pd.Timestamp("2026-10-23")
        w2 = pd.Timestamp("2026-10-30")
        rows = []
        # Week 1: tenors 1..5 map to Jan..May
        for tenor, cid in enumerate(["2027-01", "2027-02", "2027-03", "2027-04", "2027-05"], start=1):
            rows.append({"date": w1, "tenor": tenor, "contract": cid, "close": contract_prices[cid]})
        # Week 2: front expires; tenors 1..5 shift to Feb..Jun with zero true price change per contract
        for tenor, cid in enumerate(["2027-02", "2027-03", "2027-04", "2027-05", "2027-06"], start=1):
            rows.append({"date": w2, "tenor": tenor, "contract": cid, "close": contract_prices[cid]})

        sub = pd.DataFrame(rows)
        df, matched_m3, total_trans = compute_contract_matched_returns(sub)

        self.assertEqual(matched_m3, 1)
        self.assertEqual(total_trans, 1)
        self.assertAlmostEqual(float(df.loc[w2, "ret_m3"]), 0.0, places=10)
        self.assertGreater(float(df.loc[w2, "ret_m3_naive"]), 0.015)

    def test_point_in_time_publication_lags(self) -> None:
        """
        Verify Tuesday COT -> next Monday (+6d) session (since Friday 15:30 ET CFTC release
        arrives after 14:15 ET CME grain/oilseed settlement) and monthly SST -> MonthEnd + 10d lag.
        """
        tuesday = pd.Series([pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-27")])
        avail = compute_cot_tradeable_availability(tuesday)
        self.assertEqual(avail.iloc[0], pd.Timestamp("2026-10-26"))
        self.assertEqual(avail.iloc[1], pd.Timestamp("2026-11-02"))

        sst_avail = compute_monthly_sst_availability(2026, 9, lag_days=10)
        self.assertEqual(sst_avail, pd.Timestamp("2026-10-10"))

    def test_strict_walkforward_target_end_date_purge(self) -> None:
        """Ensure any observation whose 13-week target ends after `decision_date` is purged."""
        decision_date = pd.Timestamp("2026-10-30")
        df = pd.DataFrame({
            "date": [
                pd.Timestamp("2026-07-24"),  # 13w target ends 2026-10-23 <= decision_date (KEEP)
                pd.Timestamp("2026-07-31"),  # 13w target ends 2026-10-30 <= decision_date (KEEP)
                pd.Timestamp("2026-08-07"),  # 13w target ends 2026-11-06 >  decision_date (PURGE)
                pd.Timestamp("2026-10-16"),  # 13w target ends 2027-01-15 >  decision_date (PURGE)
            ],
            "fwd_13w_end_date": [
                pd.Timestamp("2026-10-23"),
                pd.Timestamp("2026-10-30"),
                pd.Timestamp("2026-11-06"),
                pd.Timestamp("2027-01-15"),
            ],
            "fwd_13w": [0.04, -0.02, 0.08, 0.01],
        })
        purged = select_purged_training_rows(df, decision_date)
        self.assertEqual(len(purged), 2)
        self.assertTrue((purged["fwd_13w_end_date"] <= decision_date).all())

    def test_black76_put_call_parity_and_greeks(self) -> None:
        """Check European Black-76 put-call parity, delta bounds, positive vega, and negative theta."""
        F, K, T, sigma, r = 66.25, 70.0, 113.0 / 365.0, 0.211, 0.0403
        call = black76_option(F=F, K=K, T=T, sigma=sigma, r=r, is_call=True)
        put = black76_option(F=F, K=K, T=T, sigma=sigma, r=r, is_call=False)

        parity_lhs = call["price"] - put["price"]
        parity_rhs = math.exp(-r * T) * (F - K)
        self.assertAlmostEqual(parity_lhs, parity_rhs, places=10)
        self.assertGreater(call["delta"], 0.0)
        self.assertLess(call["delta"], 1.0)
        self.assertGreater(call["vega_1pt"], 0.0)
        self.assertLess(call["theta_90d"], 0.0)

    def test_two_step_ivr_option_filter(self) -> None:
        """
        Verify Step 1 (<=20% IVR shortlist) and Step 2 (catalyst expiry coverage):
          - Soyoil 4% IVR & SRW Wheat 16% IVR -> PASS
          - Mar-27 Soybeans 30% IVR -> REJECT at Step 1
          - Jan-27 Soybeans 18% IVR expiring Dec 24 before Jan 10 catalyst -> REJECT at Step 2
        """
        ok_zl, _ = evaluate_two_step_option_filter(4.0, expiry_covers_catalyst=True)
        ok_zw, _ = evaluate_two_step_option_filter(16.0, expiry_covers_catalyst=True)
        bad_zs_mar, msg1 = evaluate_two_step_option_filter(30.0, expiry_covers_catalyst=True)
        bad_zs_jan, msg2 = evaluate_two_step_option_filter(18.0, expiry_covers_catalyst=False)

        self.assertTrue(ok_zl)
        self.assertTrue(ok_zw)
        self.assertFalse(bad_zs_mar)
        self.assertIn("Step 1", msg1)
        self.assertFalse(bad_zs_jan)
        self.assertIn("Step 2", msg2)

    def test_mandate_allocation_invariants(self) -> None:
        """Confirm relative % mandate shares sum to 100% in every state and exclude short Robusta."""
        self.assertEqual(MANDATE_TARGET_VOL_PCT, 10.0)
        self.assertEqual(DAY1_DEPLOYED_SHARE_PCT + CASH_RESERVE_SHARE_PCT, 100)

        day1_sum = sum(s["day1_mandate_share_pct"] for s in MANDATE_SLEEVE_SHARES_PCT)
        post_sum = sum(s["post_trigger_share_pct"] for s in MANDATE_SLEEVE_SHARES_PCT)
        fallback_sum = sum(s["no_palm_fallback_share_pct"] for s in MANDATE_SLEEVE_SHARES_PCT)

        self.assertEqual(day1_sum, 100)
        self.assertEqual(post_sum, 100)
        self.assertEqual(fallback_sum, 100)

        f3 = next(s for s in MANDATE_SLEEVE_SHARES_PCT if s["id"] == "F3")
        self.assertIn("ZERO short Robusta", f3["contracts"])

    def test_newey_west_tstat_positive_drift(self) -> None:
        """Verify HAC t-statistic is positive on a noisy positive-mean series."""
        rng = np.random.default_rng(42)
        r = 0.015 + 0.03 * rng.standard_normal(120)
        self.assertGreater(newey_west_tstat(r, max_lag=3), 2.5)


if __name__ == "__main__":
    unittest.main()
