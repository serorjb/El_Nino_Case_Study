# -*- coding: utf-8 -*-
"""
Contract-matched volatility diagnostics, Black-76 option valuation, and the
two-step implied-volatility rank (`IVR <= 20%`) payoff structure selector.
"""
from __future__ import annotations

import math
from typing import Tuple

from arch import arch_model
import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import (
    ACTIVE_SLEEVE_MARKETS,
    IV_RANK_SHORTLIST_MAX_PCT,
    UNIVERSE_META,
    get_physical_sd_overlay,
)
from .data_loader import DataPaths


def _b76_price(F: float, K: float, T: float, sigma: float, r: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0) if is_call else max(K - F, 0.0)
    df = math.exp(-r * T)
    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    return df * (F * norm.cdf(d1) - K * norm.cdf(d2)) if is_call else df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_option(
    F: float, K: float, T: float, sigma: float, r: float = 0.0403, is_call: bool = True
) -> dict:
    """
    European Black-76 futures option price, Greeks, and 1d/30d/90d flat-curve
    time decay (standard screening benchmark; exchange ag options are American).
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        delta_exp = 1.0 if (is_call and F > K) else (-1.0 if (not is_call and K > F) else 0.0)
        return {
            "price": intrinsic,
            "delta": delta_exp,
            "vega_1pt": 0.0,
            "theta_1d": 0.0,
            "theta_30d": 0.0,
            "theta_90d": 0.0,
        }
    df = math.exp(-r * T)
    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    px = _b76_price(F, K, T, sigma, r, is_call)
    delta = df * norm.cdf(d1) if is_call else -df * norm.cdf(-d1)
    vega_1pt = df * F * norm.pdf(d1) * math.sqrt(T) * 0.01
    px_1d = _b76_price(F, K, max(T - 1.0 / 365.0, 0.0), sigma, r, is_call)
    px_30d = _b76_price(F, K, max(T - 30.0 / 365.0, 0.0), sigma, r, is_call)
    px_90d = _b76_price(F, K, max(T - 90.0 / 365.0, 0.0), sigma, r, is_call)
    return {
        "price": px,
        "delta": delta,
        "vega_1pt": vega_1pt,
        "theta_1d": px_1d - px,
        "theta_30d": px_30d - px,
        "theta_90d": px_90d - px,
    }


def evaluate_two_step_option_filter(
    iv_rank_pct: float | None,
    expiry_covers_catalyst: bool = True,
    max_ivr_pct: float = IV_RANK_SHORTLIST_MAX_PCT,
) -> Tuple[bool, str]:
    """
    Two-step long-option selection filter:
      - Step 1 (Shortlist): 52-week Implied Volatility Rank (`IVR`) <= `max_ivr_pct` (20.0%).
      - Step 2 (Catalyst Expiry Match): Option expiry must cover the dated physical/policy event.
    """
    if iv_rank_pct is None or not np.isfinite(iv_rank_pct):
        return False, "No verified implied volatility surface; express linearly via futures/spreads."
    if not expiry_covers_catalyst:
        return False, f"Rejected at Step 2: contract expires before dated catalyst window (despite {iv_rank_pct:.0f}% IVR)."
    if float(iv_rank_pct) > max_ivr_pct:
        return False, f"Rejected at Step 1: 52w IV Rank ({iv_rank_pct:.0f}%) exceeds {max_ivr_pct:.0f}% shortlist ceiling."
    return True, f"Passed Step 1 ({iv_rank_pct:.0f}% IVR <= {max_ivr_pct:.0f}%) and Step 2 (catalyst-matched expiry)."


def select_optimal_payoff_structure(
    name: str,
    post_ret_6m: float,
    carry_13_pct: float,
    vrp_ratio: float,
    iv_rank: float,
) -> Tuple[str, str, str]:
    """
    Map post-overlay views and verified volatility diagnostics to the executed
    5-sleeve portfolio (`F1`-`F4` linear spreads + `O1` cheap convexity), watchlist, or veto.
    """
    if name in ("Palm Oil", "Soybean Oil"):
        return (
            "F1 Lead Spread (30% Mandate Share): Long Mar-27 Palm / Short Mar-27 Soyoil + 4% IVR 70c Call Cap",
            "1_Active_Linear_Spread",
            "Starter 30% share (~3.75 Mt Oct MPOB stock survey); scale to 50% on MPOB < 3.30 Mt, exit if Q1 > 3.85 Mt. "
            "4% IVR 70c calls cap US biofuel policy spike on short Soyoil leg.",
        )
    if name == "Raw Sugar #11":
        return (
            "F2 Calendar Bull Spread (15% Mandate Share): Long Mar-27 / Short Oct-27 Raw Sugar #11",
            "1_Active_Linear_Spread",
            "El Nino hemisphere split: dry Thai/India H1 deficit vs wet 662 Mt CS Brazil H2 cane & crowded +22.4% OI.",
        )
    if name == "Arabica Coffee":
        return (
            "F3 Calendar Bull Spread (11% Mandate Share): Long May-27 / Short Sep-27 Arabica Coffee",
            "1_Active_Linear_Spread",
            "Tight 225k NY certified washed stocks & ~68c diffs ahead of 77.2M-bag H2 Brazil harvest; ZERO short Robusta.",
        )
    if name == "Soybean Meal":
        return (
            "F4 Crush Value Spread (8% Mandate Share): Long Mar-27 Soymeal / Short Mar-27 Soyoil",
            "1_Active_Linear_Spread",
            "Tightest CBOT carry (15.3% vs 50.6% Full Carry) & 50.0% Oil Share mean-reversion; 0% if house crush overlaps.",
        )
    if name == "Wheat SRW":
        vrp_txt = f"{vrp_ratio:.2f}x" if pd.notna(vrp_ratio) else "N/A"
        return (
            "O1 Cheap Tail Convexity (1% Mandate Share): May-27 SRW Wheat 630/710 Call Spread (16% IVR)",
            "2_Active_Cheap_Convexity",
            f"Passes two-step option filter (16% IVR <= 20%, RV/IV={vrp_txt}); avoids {carry_13_pct:.1f}%/yr contango bleed "
            "while targeting S. Brazil rain quality loss, Aus dry risk & -58.3k fund shorts.",
        )
    if name == "Robusta Coffee":
        return (
            "Watchlist (0% Active Today — Conditional Long Trigger; ZERO Short Robusta)",
            "3_Conditional_Watchlist",
            "Two-clock split: +19.6% spot Vietnam exports weigh now, but Jan-Mar Central Highlands dry-season irrigation stress creates Q1 upside tail.",
        )
    if name == "Soybeans":
        _, reason = evaluate_two_step_option_filter(iv_rank, expiry_covers_catalyst=True)
        return (
            "Skipped Option Candidate (0% Active Today): Mar-27 Soybeans at 30% IVR (>20% Limit)",
            "3_Conditional_Watchlist",
            f"{reason} Jan-27 (18% IVR) expires Dec 24 before Jan 10 tariff window; re-screen mid-December.",
        )
    if name in ("Corn (CBOT)", "Wheat HRW", "Milling Wheat", "White Sugar #5"):
        return (
            "Secondary Conditional Watchlist (0% Active Today)",
            "3_Conditional_Watchlist",
            f"Constructive fundamentals ({post_ret_6m:+.1f}% unconstrained score); held behind dated planting/basis triggers.",
        )
    if (pd.notna(iv_rank) and iv_rank >= 80.0) or name in ("Cocoa (NY)", "Henry Hub Gas", "Rubber TSR20"):
        veto_detail = (
            "+422 kt 2025/26 global surplus rebuilt stocks (+25 kt 26/27e & weak grindings) at 42.6% ATM IV overrides 2023/24 drought memory"
            if name == "Cocoa (NY)"
            else ("Warm El Nino winter + 94% IVR & extreme call skew" if name == "Henry Hub Gas" else "Curve 98th pctile backwardated")
        )
        return (
            "Explicit Governance Veto (0% Allocation)",
            "4_Explicit_Governance_Veto",
            f"Vetoed: {veto_detail}.",
        )
    return (
        "Unallocated (0% Active — Below Portfolio Concentration Hurdle)",
        "4_Explicit_Governance_Veto",
        f"Unconstrained score ({post_ret_6m:+.1f}%) below portfolio concentration hurdle; preserve 35% cash reserve.",
    )


def forecast_volatility_and_select_structures(
    live_df: pd.DataFrame, panel: pd.DataFrame, paths: DataPaths
) -> pd.DataFrame:
    """
    Compute contract-matched (`c3_t == c3_{t-1}`) weekly realized/EWMA volatility plus
    El Nino regime realized vol, merge verified exchange option IV/skew quotes without
    inventing proxy surfaces, and map post-overlay views to executed structures.
    """
    sx_clean = pd.DataFrame()
    if paths.iv_carry_csv.exists():
        iv_df = pd.read_csv(paths.iv_carry_csv)
        preferred_contracts = {
            ("Soybeans", "CBOT"): "Mar-27",
            ("Wheat SRW (Chicago)", "CBOT"): "May-27",
            ("Coffee arabica", "ICE US"): "Jul-27",
            ("Raw sugar No. 11", "ICE US"): "Mar-27",
        }
        sx_rows = []
        for (mkt, exch), grp in iv_df.groupby(["market", "exchange"]):
            pref = preferred_contracts.get((mkt, exch))
            if pref and (grp["focus_contract"] == pref).any():
                sx_rows.append(grp[grp["focus_contract"] == pref].iloc[0])
            else:
                sx_rows.append(grp.iloc[0])
        sx_clean = pd.DataFrame(sx_rows)

    vol_rows = []
    for _, row in live_df.iterrows():
        name = row["name"]
        meta = UNIVERSE_META[name]
        row_dt = pd.Timestamp(row["date"])
        w_hist = panel[
            (panel["name"] == name) & (panel["date"] <= row_dt) & (panel["wdate"] >= pd.Timestamp("2000-01-01"))
        ].sort_values("date")

        # Primary deferred realized vol: 100% contract-ID-matched weekly M3 returns (zero roll jumps)
        w_r = w_hist["ret_m3"].dropna() * 100.0
        cm_rv26 = float(w_r.tail(26).std() * math.sqrt(52.0)) if len(w_r) >= 13 else np.nan
        cm_ewma = (
            float(np.sqrt((w_r ** 2).ewm(alpha=0.06, adjust=False).mean().iloc[-1] * 52.0))
            if len(w_r) >= 13
            else cm_rv26
        )
        uncond_rv = float(w_r.std() * math.sqrt(52.0)) if len(w_r) >= 26 else cm_rv26

        nino_slice = w_hist[(w_hist["nino34"] >= 1.5) | (w_hist["nino34_lag26"] >= 1.5)]["ret_m3"].dropna() * 100.0
        elnino_rv = float(nino_slice.std() * math.sqrt(52.0)) if len(nino_slice) >= 25 else uncond_rv

        contract_matched_rv = 0.50 * cm_ewma + 0.50 * cm_rv26
        forecast_rv_6m = 0.75 * contract_matched_rv + 0.25 * elnino_rv

        # Secondary daily front-continuation GJR-GARCH & HAR-RV diagnostics (strictly <= row_dt)
        gjr_vol_pct = np.nan
        har_vol_pct = np.nan
        pos_leverage_gamma = np.nan
        pxd_file = paths.pxd_dir / meta["pxd"] if meta["pxd"] else None
        if pxd_file is not None and pxd_file.exists():
            pxd = pd.read_csv(pxd_file, parse_dates=["date"]).sort_values("date")
            pxd = pxd[pxd["date"] <= row_dt]
            d_ret = np.log(pxd["close"] / pxd["close"].shift(1)).dropna() * 100.0
            d_ret_clean = d_ret[d_ret.abs() < 18.0]
            if len(d_ret_clean) > 500:
                am = arch_model(-d_ret_clean, vol="Garch", p=1, o=1, q=1, dist="normal", rescale=False)
                res = am.fit(disp="off")
                fcast = res.forecast(horizon=63, reindex=False)
                gjr_vol_pct = float(np.sqrt(np.mean(fcast.variance.values[-1])) * math.sqrt(252.0))
                pos_leverage_gamma = float(res.params.get("gamma[1]", 0.0))

                rv_d = d_ret_clean ** 2
                rv5 = rv_d.rolling(5).mean() * 252.0
                rv21 = rv_d.rolling(21).mean() * 252.0
                rv63 = rv_d.rolling(63).mean() * 252.0
                rv_fwd21 = rv_d.rolling(21).mean().shift(-21) * 252.0
                har_df = pd.DataFrame({"y": rv_fwd21, "rv5": rv5, "rv21": rv21, "rv63": rv63}).dropna()
                Xh = np.column_stack([np.ones(len(har_df)), har_df[["rv5", "rv21", "rv63"]].values])
                bh = np.linalg.lstsq(Xh, har_df["y"].values, rcond=None)[0]
                cur_var = float(np.dot(bh, [1.0, rv5.iloc[-1], rv21.iloc[-1], rv63.iloc[-1]]))
                har_vol_pct = math.sqrt(max(cur_var, 9.0))

        atm_iv = np.nan
        iv_rank = np.nan
        skew = np.nan
        pct_fc = np.nan
        focus_c = str(row["c3"])
        if not sx_clean.empty:
            sx_match = sx_clean[(sx_clean["market"] == meta["market"]) & (sx_clean["exchange"] == meta["exchange"])]
            if not sx_match.empty:
                sx = sx_match.iloc[0]
                focus_c = str(sx["focus_contract"])
                src_doc = str(sx.get("source_doc", ""))
                has_verified_iv = ("IV Recap" in src_doc) or ("IV vs Price" in src_doc)
                if has_verified_iv and pd.notna(sx["atm_iv"]):
                    atm_iv = float(sx["atm_iv"])
                if has_verified_iv and pd.notna(sx["iv_rank_52w_pct"]):
                    iv_rank = float(sx["iv_rank_52w_pct"])
                if has_verified_iv and pd.notna(sx["skew_2otm_call_minus_put"]):
                    skew = float(sx["skew_2otm_call_minus_put"])
                if pd.notna(sx["pct_full_carry"]):
                    pct_fc = float(sx["pct_full_carry"])

        vrp_ratio = (forecast_rv_6m / atm_iv) if (pd.notna(atm_iv) and atm_iv > 0) else np.nan

        prior_pct = float(row["exp_ret_6m_pct"])
        overlay_adj_pct, overlay_rationale = get_physical_sd_overlay(name)
        unconstrained_post_pct = round(prior_pct + overlay_adj_pct, 2)
        active_post_rule_pct = unconstrained_post_pct if name in ACTIVE_SLEEVE_MARKETS else 0.0

        structure, role_bucket, rationale = select_optimal_payoff_structure(
            name=name,
            post_ret_6m=unconstrained_post_pct,
            carry_13_pct=100.0 * float(row["carry_13"]),
            vrp_ratio=vrp_ratio,
            iv_rank=iv_rank,
        )

        vol_rows.append({
            "name": name,
            "sector": meta["sector"],
            "focus_contract": focus_c,
            "prior_stat_ret_6m_pct": prior_pct,
            "physical_sd_adj_pct": overlay_adj_pct,
            "unconstrained_posterior_6m_pct": unconstrained_post_pct,
            "active_post_rule_view_6m_pct": active_post_rule_pct,
            "alpha_z": round(float(row["alpha_z"]), 2),
            "carry_13_ann_pct": round(100.0 * float(row["carry_13"]), 2),
            "pct_full_carry": pct_fc if pd.notna(pct_fc) else None,
            "rv_spread_z": round(float(row["rv_spread_z"]), 2),
            "cot_z": round(float(row["cot_z"]), 2),
            "contract_matched_rv26_pct": round(cm_rv26, 1) if pd.notna(cm_rv26) else None,
            "contract_matched_ewma_pct": round(cm_ewma, 1) if pd.notna(cm_ewma) else None,
            "gjr_garch_vol_pct": round(gjr_vol_pct, 1) if pd.notna(gjr_vol_pct) else None,
            "har_rv_vol_pct": round(har_vol_pct, 1) if pd.notna(har_vol_pct) else None,
            "elnino_regime_rv_pct": round(elnino_rv, 1) if pd.notna(elnino_rv) else None,
            "forecast_rv_6m_pct": round(forecast_rv_6m, 1) if pd.notna(forecast_rv_6m) else None,
            "atm_iv_pct": round(atm_iv, 1) if pd.notna(atm_iv) else None,
            "iv_rank_52w_pct": round(iv_rank, 0) if pd.notna(iv_rank) else None,
            "skew_2otm": round(skew, 1) if pd.notna(skew) else None,
            "vol_edge_ratio_rv_iv": round(vrp_ratio, 2) if pd.notna(vrp_ratio) else None,
            "pos_spike_garch_gamma": round(pos_leverage_gamma, 3) if pd.notna(pos_leverage_gamma) else None,
            "selected_structure": structure,
            "portfolio_role": role_bucket,
            "physical_sd_rationale": overlay_rationale,
            "selector_rationale": rationale,
        })

    return pd.DataFrame(vol_rows).sort_values(
        ["portfolio_role", "active_post_rule_view_6m_pct", "unconstrained_posterior_6m_pct"],
        ascending=[True, False, False],
    )
