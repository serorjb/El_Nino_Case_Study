#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end CLI runner for the Quantamental Cross-Curve Commodity Screening &
Volatility-Mandate Portfolio Engine.

Example usage:
    python run_pipeline.py --data-dir ./data --output-dir ./output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import pandas as pd

from quantamental import (
    DataPaths,
    add_cross_commodity_and_macro_features,
    build_mandate_portfolio_and_scenarios,
    build_weekly_market_panel,
    forecast_volatility_and_select_structures,
    load_macro_fx_enso_cot,
    run_purged_walkforward,
)

warnings.filterwarnings("ignore")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Quantamental Cross-Curve Commodity & ENSO Portfolio Engine."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to input market, macro FX, COT, and ENSO data directory (default: ./data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where summary JSON and diagnostic CSVs will be written (default: ./output).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = DataPaths.from_dir(args.data_dir)
    if not paths.curves_long_csv.exists():
        sys.stderr.write(
            f"[ERROR] Weekly multi-tenor curve file not found under: {paths.root}\n"
            "Provide `--data-dir /path/to/data` following the schema documented in README.md.\n"
        )
        return 1

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("QUANTAMENTAL CROSS-CURVE COMMODITY & ENSO PORTFOLIO ENGINE")
    print("=" * 84)

    print("[1/5] Loading point-in-time FX, NOAA ENSO (+10d lag), and post-settlement CFTC COT...")
    ext = load_macro_fx_enso_cot(paths)
    long_df = pd.read_csv(paths.curves_long_csv, parse_dates=["date"])
    panel, roll_audit = build_weekly_market_panel(long_df, ext)

    print("[2/5] Building exact-date cross-commodity processing margins & lagged ENSO features...")
    panel, spread_snap = add_cross_commodity_and_macro_features(panel, long_df, ext)

    print("[3/5] Running target-end-date purged (h=13w) walk-forward ablation (2006-2026)...")
    bt_summary, bt_curves, live_df = run_purged_walkforward(panel)

    print("[4/5] Evaluating contract-matched RV, S&D overrides, and two-step <=20% IVR filter...")
    signals_df = forecast_volatility_and_select_structures(live_df, panel, paths)

    print("[5/5] Computing 10.0%-vol mandate risk shares, FX sensitivity, and joint scenarios...")
    portfolio_summary = build_mandate_portfolio_and_scenarios(long_df, ext)

    roll_audit.to_csv(out_dir / "roll_audit.csv", index=False)
    bt_curves.to_csv(out_dir / "walkforward_curves.csv")
    signals_df.to_csv(out_dir / "signals.csv", index=False)

    full_summary = {
        "methodology_hygiene": {
            "excess_returns": "Exact YYYY-MM contract-ID matched (`c_t == c_{t-1}`) across tenors M1..M4 (zero continuation roll jumps)",
            "fx_pre2014_alignment": "Monthly BRL & DXY backfills shifted from YYYY-MM-01 to MonthEnd(0) before backward merge",
            "cot_publication_lag": "CFTC Tuesday cutoff released Friday 15:30 ET (after 14:15 ET CME settle) -> enters next Monday session (+6d)",
            "enso_publication_lag": "+10-day operational release lag after month-end on monthly SST history; daily 5km / weekly OISST live",
            "cross_commodity_alignment": "Exact market observation `date` merge (no cross-day Friday `wdate` peeking)",
            "walkforward_purge": "Strict target-end-date purge (`fwd_13w_end_date <= decision_date`) across 20 continuous markets (2006-2026 weekly settlement diagnostic)",
            "option_filter": "Two-step gate: Step 1 requires 52w IV Rank <= 20%; Step 2 requires expiry covering dated physical catalyst",
            "portfolio_sizing": "Relative % of a 10.0% Annualized Volatility Mandate (65% vol budget deployed Day 1 = 6.5% expected vol; 35% unallocated risk reserve)",
        },
        "spread_snapshot_oct29": spread_snap,
        "cot_latest_snapshot": ext["cot_latest_snapshot"],
        "walkforward_ablation": bt_summary,
        "portfolio": portfolio_summary,
    }
    with open(out_dir / "framework_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    print("\n--- Walk-Forward Factor Ablation (2006-2026, Purged h=13w, 20 Markets) ---")
    for k in (
        "Benchmark_EW",
        "M1_ENSO_Only",
        "M2_Curve_Val_Mom",
        "M3_Curve_RV_COT",
        "M4_Full_MultiFactor",
        "M5_BL_VolTarget_Net8bps",
    ):
        st = bt_summary[k]
        print(
            f"  {k:26s} | Ret: {st['ann_excess_return_pct']:+5.2f}% | Vol: {st['ann_vol_pct']:5.2f}% "
            f"| Sharpe: {st['sharpe']:+5.2f} (NW t={st['newey_west_tstat']:+4.2f}) | MaxDD: {st['max_drawdown_pct']:6.2f}%"
        )
    ic_info = bt_summary["rank_ic"]
    print(f"  Out-of-Sample Rank IC (M4): {ic_info['m4_full_mean_ic']:+.3f} (NW t={ic_info['m4_ic_tstat']:+.2f})")

    mandate = portfolio_summary["mandate"]
    cov_d1 = portfolio_summary["covariance_risk_decomposition"]["day1_ndf_fx_hedged"]
    opt = portfolio_summary["options_audit_pct_of_mandate"]
    scen = portfolio_summary["scenario_analysis"]

    print("\n--- 10.0% Annualized Volatility Mandate Allocation (Relative % Sizing) ---")
    for sl in mandate["sleeve_allocations"]:
        print(
            f"  {sl['id']:4s} | Day-1: {sl['day1_mandate_share_pct']:2d}% "
            f"(Post-Trigger: {sl['post_trigger_share_pct']:2d}%) | {sl['sleeve']}"
        )
    print(
        f"  Expected Portfolio Vol: {mandate['day1_expected_vol_pct']:.1f}% Day-1 "
        f"({mandate['post_trigger_expected_vol_pct']:.1f}% Post-Trigger) | "
        f"Diversification Benefit: {cov_d1['diversification_benefit_pct']:.1f}%"
    )
    print(
        f"  Option Spend: {opt['total_day1_option_premium_pct']:.2f}% of mandate "
        f"(90d flat-curve decay: {opt['flat_curve_90d_theta_bleed_pct']:+.2f}%)"
    )

    print("\n--- Probability-Weighted Pre-Roll Return Scenarios (% of Mandate; 0% Cash Yield) ---")
    for sc in scen["scenarios"]:
        print(
            f"  [{sc['probability_pct']:2d}% prob] {sc['scenario']:38s} -> "
            f"Day-1: {sc['day1_return_pct']:+5.1f}% (Post-Trigger: {sc['post_trigger_return_pct']:+5.1f}%)"
        )
    print(f"  Combined Probability-Weighted Scenario Mean: {scen['probability_weighted_return_pct']:+.2f}%")
    print(f"\nOutputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
