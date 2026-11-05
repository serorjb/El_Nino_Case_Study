# -*- coding: utf-8 -*-
"""
Relative volatility-mandate portfolio sizing (10.0% annualized vol target),
104-week contract-matched covariance risk decomposition, USD/MYR FX convention
sensitivity, Black-76 option cost audit, and probability-weighted scenarios.

All allocations, risk metrics, option costs, and stress returns are expressed
strictly in dimensionless `%` relative to the 10.0% Annualized Volatility Mandate
(zero dollar book-size assumptions; zero cash interest included in returns).
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    CASH_RESERVE_SHARE_PCT,
    DAY1_DEPLOYED_SHARE_PCT,
    MANDATE_TARGET_VOL_PCT,
)
from .volatility import black76_option


# Unit-mandate leg notional weights (dimensionless fraction of a 10.0%-vol mandate)
UNIT_MANDATE_WEIGHTS: Dict[str, float] = {
    "F1_fcpo_full": 0.350638,     # +35.06% post-trigger; +17.53% Day-1 starter (0.5x)
    "F1_zl_full": -0.424000,      # -42.40% post-trigger; -21.20% Day-1 starter (capped 1:1 by 70c calls)
    "F2_sb_mar": 0.399784,        # +39.98% Mar-27 Raw Sugar #11
    "F2_sb_oct": -0.378019,       # -37.80% Oct-27 Raw Sugar #11 (equal physical quantity)
    "F3_kc_may": 0.614250,        # +61.43% May-27 Arabica Coffee (ZERO short Robusta)
    "F3_kc_sep": -0.601988,       # -60.20% Sep-27 Arabica Coffee (equal physical quantity)
    "F4_zm_mar": 0.055183,        # +5.52% Mar-27 Soybean Meal
    "F4_zl_mar": -0.053000,       # -5.30% Mar-27 Soybean Oil (uncapped; audited jointly with F1)
    "O1_zw_may_und": 0.155688,    # +15.57% May-27 SRW Wheat 630/710 Call Spread underlying notional
}

MANDATE_SLEEVE_SHARES_PCT: List[dict] = [
    {
        "id": "F1",
        "sleeve": "Palm Oil vs. Soyoil + 70c Call Cap",
        "contracts": "Long Mar-27 FCPO / Short Mar-27 ZL + 70c ZL Call",
        "day1_mandate_share_pct": 30,
        "post_trigger_share_pct": 50,
        "no_palm_fallback_share_pct": 0,
        "scale_up_condition": "MPOB stocks draw < 3.30 Mt (from ~3.75 Mt Oct survey) & Mar-27 POBO < -180 USD/t",
        "exit_stop_condition": "Q1 MPOB stocks > 3.85 Mt or Indonesia delays B40 biodiesel blending",
    },
    {
        "id": "F2",
        "sleeve": "Raw Sugar #11 Mar/Oct Bull Spread",
        "contracts": "Long Mar-27 SBH7 / Short Oct-27 SBV7 (+1.06 c/lb)",
        "day1_mandate_share_pct": 15,
        "post_trigger_share_pct": 15,
        "no_palm_fallback_share_pct": 25,
        "scale_up_condition": "Hold through Q1 Thai/India crush tally",
        "exit_stop_condition": "Mar/Oct spread breaks < +0.35 c/lb",
    },
    {
        "id": "F3",
        "sleeve": "Arabica Coffee May/Sep Bull Spread",
        "contracts": "Long May-27 KCK7 / Short Sep-27 KCU7 (+5.45 c/lb; ZERO short Robusta)",
        "day1_mandate_share_pct": 11,
        "post_trigger_share_pct": 11,
        "no_palm_fallback_share_pct": 25,
        "scale_up_condition": "Optional Robusta long trigger if Jan-Mar Vietnam rainfall < 50% normal (25% combined coffee cap)",
        "exit_stop_condition": "NY certified washed stocks rebuild > 450k bags or May/Sep < +0.50 c/lb",
    },
    {
        "id": "F4",
        "sleeve": "Soymeal vs. Soyoil Crush Value Spread",
        "contracts": "Long Mar-27 ZMH7 / Short Mar-27 ZLH7",
        "day1_mandate_share_pct": 8,
        "post_trigger_share_pct": 8,
        "no_palm_fallback_share_pct": 10,
        "scale_up_condition": "0% if commercial crush book already carries short Oil Share",
        "exit_stop_condition": "CBOT Oil Share breaks > 53.5%",
    },
    {
        "id": "O1",
        "sleeve": "SRW Wheat 630/710 Call Spread (16% IVR)",
        "contracts": "Long May-27 ZWK7 630c / Short 710c Call Spread",
        "day1_mandate_share_pct": 1,
        "post_trigger_share_pct": 1,
        "no_palm_fallback_share_pct": 5,
        "scale_up_condition": "Defined-loss premium sleeve (0.58% of mandate)",
        "exit_stop_condition": "Premium-capped (no variation margin tail); monetize >= 705 c/bu",
    },
    {
        "id": "CASH",
        "sleeve": "Unallocated Risk Reserve",
        "contracts": "Dry powder reserved for dated physical/weather triggers",
        "day1_mandate_share_pct": CASH_RESERVE_SHARE_PCT,
        "post_trigger_share_pct": 15,
        "no_palm_fallback_share_pct": 35,
        "scale_up_condition": "Deploys +20% into F1 on MPOB draw < 3.30 Mt (or Jan-Mar VN Robusta / late-Feb BR Corn triggers)",
        "exit_stop_condition": "Zero cash interest included in return attribution",
    },
]


def matched_simple_returns_from_long(
    long_df: pd.DataFrame, market: str, exchange: str, slot: int
) -> pd.Series:
    """
    Compute exact contract-ID-matched (`c_{t-1} == c_t`) weekly simple excess returns
    for tenor `slot` directly from the multi-tenor curve dataset.
    """
    sub = long_df[
        (long_df["market"] == market)
        & (long_df["exchange"] == exchange)
        & (~long_df["unit"].str.contains("unconfirmed", na=False))
    ].copy()
    piv_p = sub.pivot_table(index="date", columns="tenor", values="close", aggfunc="last").sort_index()
    piv_c = sub.pivot_table(index="date", columns="tenor", values="contract", aggfunc="last").sort_index()
    dc_lookup = sub.set_index(["date", "contract"])["close"].to_dict()
    dates = list(piv_p.index)

    out = {}
    for k in range(1, len(dates)):
        c_prev = piv_c.iloc[k - 1].get(slot, np.nan)
        p_prev = piv_p.iloc[k - 1].get(slot, np.nan)
        if pd.notna(c_prev) and pd.notna(p_prev) and p_prev > 0:
            p_now = dc_lookup.get((dates[k], c_prev), np.nan)
            if pd.notna(p_now) and p_now > 0:
                out[dates[k]] = (p_now / p_prev) - 1.0
    return pd.Series(out, name=f"{market}_m{slot}")


def compute_cross_market_correlation(long_df: pd.DataFrame, weeks: int = 104) -> dict:
    """Compute 104-week contract-matched M3 correlation matrix across 11 core markets."""
    corr_markets = [
        ("Palm Oil", "Crude palm oil", "Bursa Malaysia"),
        ("Soyoil", "Soybean oil", "CBOT"),
        ("Soymeal", "Soybean meal", "CBOT"),
        ("Soybeans", "Soybeans", "CBOT"),
        ("Corn", "Corn", "CBOT"),
        ("Wheat SRW", "Wheat SRW (Chicago)", "CBOT"),
        ("Arabica", "Coffee arabica", "ICE US"),
        ("Robusta", "Coffee robusta", "ICE Europe"),
        ("Raw Sugar", "Raw sugar No. 11", "ICE US"),
        ("Cocoa NY", "Cocoa (New York)", "ICE US"),
        ("Gasoil", "Low sulphur gasoil", "ICE Europe"),
    ]
    c_series = {}
    for label, mkt, exch in corr_markets:
        s = matched_simple_returns_from_long(long_df, mkt, exch, 3)
        s.index = s.index.to_period("W-FRI").end_time.normalize()
        c_series[label] = s.groupby(level=0).last()
    corr_df = pd.DataFrame(c_series).tail(weeks).corr().round(2)
    return {
        "labels": [x[0] for x in corr_markets],
        "values": corr_df.values.tolist(),
    }


def build_mandate_portfolio_and_scenarios(long_df: pd.DataFrame, ext: dict) -> dict:
    """
    Construct the relative 10.0%-volatility mandate portfolio, 104-week contract-matched
    covariance decomposition, USD/MYR FX sensitivity, Black-76 option cost audit, and
    5 probability-weighted 6-month return scenarios.
    """
    r_sofr = 0.0403
    fx_df = ext["usdmyr"].set_index("date").sort_index()
    fx_oct29 = float(fx_df.loc[:pd.Timestamp("2026-10-29"), "usdmyr"].iloc[-1])

    # Exact Oct 29, 2026 contract settlement prices
    px = {
        "FCPOH7_myr": 4773.0,
        "ZLH7_c": 66.25,
        "ZMH7_st": 331.10,
        "SBH7_c": 19.47,
        "SBV7_c": 18.41,
        "KCK7_c": 273.00,
        "KCU7_c": 267.55,
        "ZWK7_c": 622.75,
    }

    # Verified Black-76 option valuations
    zl_cap = black76_option(F=px["ZLH7_c"], K=70.0, T=113.0 / 365.0, sigma=0.211, r=r_sofr, is_call=True)
    zw_c630 = black76_option(F=px["ZWK7_c"], K=630.0, T=175.0 / 365.0, sigma=0.249, r=r_sofr, is_call=True)
    zw_c710 = black76_option(F=px["ZWK7_c"], K=710.0, T=175.0 / 365.0, sigma=0.257, r=r_sofr, is_call=True)
    zw_debit_c = zw_c630["price"] - zw_c710["price"]
    zw_cs_delta = zw_c630["delta"] - zw_c710["delta"]

    # 104-week contract-matched return panel
    ret_map = {
        "fcpo": matched_simple_returns_from_long(long_df, "Crude palm oil", "Bursa Malaysia", 3),
        "zl": matched_simple_returns_from_long(long_df, "Soybean oil", "CBOT", 3),
        "zm": matched_simple_returns_from_long(long_df, "Soybean meal", "CBOT", 3),
        "sb_h": matched_simple_returns_from_long(long_df, "Raw sugar No. 11", "ICE US", 1),
        "sb_v": matched_simple_returns_from_long(long_df, "Raw sugar No. 11", "ICE US", 4),
        "kc_k": matched_simple_returns_from_long(long_df, "Coffee arabica", "ICE US", 3),
        "kc_u": matched_simple_returns_from_long(long_df, "Coffee arabica", "ICE US", 5),
        "zw": matched_simple_returns_from_long(long_df, "Wheat SRW (Chicago)", "CBOT", 3),
    }
    ret_df = pd.DataFrame(ret_map).dropna(how="all").sort_index().tail(104)
    r_sb_m3 = matched_simple_returns_from_long(long_df, "Raw sugar No. 11", "ICE US", 3).reindex(ret_df.index)
    ret_df["sb_h"] = ret_df["sb_h"].fillna(r_sb_m3)
    ret_df = ret_df.dropna(how="any")

    fx_weekly = fx_df["usdmyr"].reindex(fx_df.index.union(ret_df.index)).sort_index().ffill().bfill().reindex(ret_df.index)
    r_fx = fx_weekly.pct_change().fillna(0.0).values

    r_fcpo_local = ret_df["fcpo"].values
    r_fcpo_usd = (1.0 + r_fcpo_local) / (1.0 + r_fx) - 1.0
    r_fcpo_vm = r_fcpo_local / np.maximum(1.0 + r_fx, 0.5)

    w = UNIT_MANDATE_WEIGHTS
    w_zl_f1_delta_full = w["F1_zl_full"] * (1.0 - zl_cap["delta"])
    w_zw_delta = w["O1_zw_may_und"] * zw_cs_delta

    # Dimensionless weekly sleeve returns (% of unit mandate = 100 * r_sleeve)
    r_f1_full_usd = w["F1_fcpo_full"] * r_fcpo_usd + w_zl_f1_delta_full * ret_df["zl"].values
    r_f1_full_vm = w["F1_fcpo_full"] * r_fcpo_vm + w_zl_f1_delta_full * ret_df["zl"].values
    r_f2 = w["F2_sb_mar"] * ret_df["sb_h"].values + w["F2_sb_oct"] * ret_df["sb_v"].values
    r_f3 = w["F3_kc_may"] * ret_df["kc_k"].values + w["F3_kc_sep"] * ret_df["kc_u"].values
    r_f4 = w["F4_zm_mar"] * ret_df["zm"].values + w["F4_zl_mar"] * ret_df["zl"].values
    r_o1 = w_zw_delta * ret_df["zw"].values

    sleeves = ["F1", "F2", "F3", "F4", "O1"]

    def _decomp_pct(f1_scale: float, use_vm: bool = False, shrink: float = 0.25) -> dict:
        f1_s = r_f1_full_vm if use_vm else r_f1_full_usd
        mat_pct = 100.0 * np.column_stack([f1_scale * f1_s, r_f2, r_f3, r_f4, r_o1])
        cov_raw = np.cov(mat_pct, rowvar=False, ddof=1) * 52.0
        cov_s = (1.0 - shrink) * cov_raw + shrink * np.diag(np.diag(cov_raw))
        ones = np.ones(len(sleeves))
        p_var = float(ones @ cov_s @ ones)
        p_vol_pct = math.sqrt(max(p_var, 1e-8))
        deployed_crc_pct = 100.0 * (cov_s @ ones) / max(p_var, 1e-8)
        st_vol_pct = np.sqrt(np.maximum(np.diag(cov_s), 0.0))
        corr_f1_f4 = float(cov_s[0, 3] / max(st_vol_pct[0] * st_vol_pct[3], 1e-8))
        return {
            "expected_ann_vol_pct": round(p_vol_pct, 1),
            "sum_standalone_vol_pct": round(float(np.sum(st_vol_pct)), 1),
            "diversification_benefit_pct": round(100.0 * (1.0 - p_vol_pct / max(float(np.sum(st_vol_pct)), 1e-8)), 1),
            "f1_f4_return_corr": round(corr_f1_f4, 2),
            "deployed_crc_share_pct": {s: round(float(c), 1) for s, c in zip(sleeves, deployed_crc_pct)},
            "standalone_vol_pct": {s: round(float(v), 2) for s, v in zip(sleeves, st_vol_pct)},
        }

    d1_usd = _decomp_pct(0.50, use_vm=False)
    pg_usd = _decomp_pct(1.00, use_vm=False)
    d1_vm = _decomp_pct(0.50, use_vm=True)

    # Option premium & 90-day flat-curve decay (% of unit mandate)
    f1_day1_zl_abs = 0.50 * abs(w["F1_zl_full"])
    f1_call_prem_day1_pct = 100.0 * f1_day1_zl_abs * (zl_cap["price"] / px["ZLH7_c"])
    f1_call_prem_gate_pct = 2.0 * f1_call_prem_day1_pct
    o1_prem_pct = 100.0 * w["O1_zw_may_und"] * (zw_debit_c / px["ZWK7_c"])

    f1_theta90_day1_pct = 100.0 * f1_day1_zl_abs * (zl_cap["theta_90d"] / px["ZLH7_c"])
    o1_theta90_pct = 100.0 * w["O1_zw_may_und"] * ((zw_c630["theta_90d"] - zw_c710["theta_90d"]) / px["ZWK7_c"])

    # 5 Probability-Weighted 6M Return Scenarios (% of Mandate; zero cash interest assumed)
    def _scenario_ret_pct(
        name: str,
        prob_pct: int,
        drivers: str,
        d_fcpo_myr: float,
        d_zl_c: float,
        d_zm_st: float,
        d_sb_sp_c: float,
        d_kc_sp_c: float,
        d_zw_c: float,
        d_iv_pts: float,
        days: int,
    ) -> dict:
        rem_zl = max((113.0 - days) / 365.0, 1e-4)
        rem_zw = max((175.0 - days) / 365.0, 1e-4)

        f1_lin_d1 = (
            0.50 * w["F1_fcpo_full"] * (d_fcpo_myr / px["FCPOH7_myr"])
            + 0.50 * w["F1_zl_full"] * (d_zl_c / px["ZLH7_c"])
        )
        zl_c_new = black76_option(
            px["ZLH7_c"] + d_zl_c, 70.0, rem_zl, max(0.211 + d_iv_pts * 0.01, 0.10), r_sofr, True
        )["price"]
        f1_opt_d1 = f1_day1_zl_abs * ((zl_c_new - zl_cap["price"]) / px["ZLH7_c"])
        f1_d1_pct = 100.0 * (f1_lin_d1 + f1_opt_d1)
        f1_gate_pct = 2.0 * f1_d1_pct

        f2_pct = 100.0 * w["F2_sb_mar"] * (d_sb_sp_c / px["SBH7_c"])
        f3_pct = 100.0 * w["F3_kc_may"] * (d_kc_sp_c / px["KCK7_c"])
        f4_pct = 100.0 * (
            w["F4_zm_mar"] * (d_zm_st / px["ZMH7_st"]) + w["F4_zl_mar"] * (d_zl_c / px["ZLH7_c"])
        )

        zw_new = (
            black76_option(px["ZWK7_c"] + d_zw_c, 630.0, rem_zw, max(0.249 + d_iv_pts * 0.01, 0.14), r_sofr, True)["price"]
            - black76_option(px["ZWK7_c"] + d_zw_c, 710.0, rem_zw, max(0.257 + d_iv_pts * 0.01, 0.14), r_sofr, True)["price"]
        )
        o1_pct = 100.0 * w["O1_zw_may_und"] * ((zw_new - zw_debit_c) / px["ZWK7_c"])

        tot_d1_pct = f1_d1_pct + f2_pct + f3_pct + f4_pct + o1_pct
        tot_gate_pct = f1_gate_pct + f2_pct + f3_pct + f4_pct + o1_pct
        return {
            "scenario": name,
            "probability_pct": prob_pct,
            "day1_return_pct": round(tot_d1_pct, 1),
            "day1_return_exact_pct": round(tot_d1_pct, 4),
            "post_trigger_return_pct": round(tot_gate_pct, 1),
            "sleeve_returns_day1_pct": {
                "F1": round(f1_d1_pct, 2),
                "F2": round(f2_pct, 2),
                "F3": round(f3_pct, 2),
                "F4": round(f4_pct, 2),
                "O1": round(o1_pct, 2),
            },
            "drivers": drivers,
        }

    scenarios = [
        _scenario_ret_pct(
            "Severe El Nino + Quality Stress", 20,
            "Palm lag + B40 tighten; H1 Sugar & Arabica certs squeeze",
            +650.0, -3.5, +28.0, +1.10, +8.5, +75.0, +4.0, 90,
        ),
        _scenario_ret_pct(
            "Base El Nino Tightening", 40,
            "POBO converges +147 USD/t (to -145 USD/t); H1/H2 Sugar & Meal hold",
            +420.0, -2.0, +18.0, +0.65, +5.0, +45.0, +2.0, 90,
        ),
        _scenario_ret_pct(
            "Unchanged Curves (90d Option Decay)", 15,
            "Futures spreads flat (0%); 90-day option time decay only",
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 90,
        ),
        _scenario_ret_pct(
            "US Biofuel Policy Spike (ZL +12c)", 10,
            "EPA 45Z spikes Soyoil +12c; F1 70c calls cap loss, F4 linear",
            +180.0, +12.0, -14.0, +0.10, 0.0, +10.0, +5.0, 60,
        ),
        _scenario_ret_pct(
            "Benign Weather / Early La Nina", 15,
            "SE Asia rains recover; palm stocks build; spreads unwind",
            -380.0, +3.0, -20.0, -0.55, -5.0, -40.0, -3.0, 90,
        ),
    ]
    prob_weighted_ret = round(
        sum(s["probability_pct"] * s["day1_return_exact_pct"] / 100.0 for s in scenarios), 2
    )
    for s in scenarios:
        s.pop("day1_return_exact_pct", None)

    return {
        "as_of_date": "2026-10-29",
        "fx_usdmyr": round(fx_oct29, 4),
        "mandate": {
            "full_target_vol_pct": MANDATE_TARGET_VOL_PCT,
            "day1_deployed_share_pct": DAY1_DEPLOYED_SHARE_PCT,
            "cash_reserve_share_pct": CASH_RESERVE_SHARE_PCT,
            "day1_expected_vol_pct": d1_usd["expected_ann_vol_pct"],
            "post_trigger_expected_vol_pct": pg_usd["expected_ann_vol_pct"],
            "sleeve_allocations": MANDATE_SLEEVE_SHARES_PCT,
        },
        "options_audit_pct_of_mandate": {
            "f1_zl_70c_call_cap_day1_pct": round(f1_call_prem_day1_pct, 2),
            "f1_zl_70c_call_cap_post_trigger_pct": round(f1_call_prem_gate_pct, 2),
            "o1_srw_630_710_call_spread_pct": round(o1_prem_pct, 2),
            "total_day1_option_premium_pct": round(f1_call_prem_day1_pct + o1_prem_pct, 2),
            "flat_curve_90d_theta_bleed_pct": round(f1_theta90_day1_pct + o1_theta90_pct, 2),
            "zl_cap_delta": round(zl_cap["delta"], 3),
            "zw_call_spread_net_delta": round(zw_cs_delta, 3),
        },
        "shared_short_soyoil_audit_pct": {
            "day1_f1_capped_short_weight_pct": round(100.0 * 0.50 * w["F1_zl_full"], 2),
            "day1_f4_uncapped_short_weight_pct": round(100.0 * w["F4_zl_mar"], 2),
            "day1_total_short_weight_pct": round(100.0 * (0.50 * w["F1_zl_full"] + w["F4_zl_mar"]), 2),
            "capped_share_of_short_soyoil_pct": 80.0,
        },
        "covariance_risk_decomposition": {
            "complete_matched_weeks": int(len(ret_df)),
            "day1_ndf_fx_hedged": d1_usd,
            "post_trigger_ndf_fx_hedged": pg_usd,
            "day1_unfunded_myr_variation_margin": d1_vm,
            "fx_convention_f1_crc_diff_pp": round(
                abs(d1_usd["deployed_crc_share_pct"]["F1"] - d1_vm["deployed_crc_share_pct"]["F1"]), 1
            ),
        },
        "scenario_analysis": {
            "evaluation_horizon": "60-90d pre-roll horizon (prior to mid-Feb Mar-27 First Notice Day roll)",
            "cash_interest_assumed_pct": 0.0,
            "probability_weighted_return_pct": prob_weighted_ret,
            "scenarios": scenarios,
        },
        "cross_market_correlation_104w": compute_cross_market_correlation(long_df),
    }
