# -*- coding: utf-8 -*-
"""
Point-in-time market curve, macro FX, CFTC COT, and NOAA ENSO loaders, plus
exact YYYY-MM contract-ID-matched return and carry panel construction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import SST_MONTHLY_RELEASE_LAG_DAYS, UNIVERSE_META


@dataclass(frozen=True)
class DataPaths:
    """Standardized input file locations resolved relative to a local `data/` directory."""

    root: Path
    curves_long_csv: Path
    usdmyr_csv: Path
    eurusd_csv: Path
    brl_monthly_csv: Path
    brl_daily_csv: Path
    dxy_monthly_csv: Path
    dxy_daily_csv: Path
    nino_txt: Path
    cot_history_csv: Path
    iv_carry_csv: Path
    pxd_dir: Path

    @classmethod
    def from_dir(cls, data_dir: Path) -> "DataPaths":
        d = Path(data_dir).resolve()
        # Resolve weekly multi-tenor curves file cleanly across standard or nested layouts
        candidates = [d / "curves" / "weekly_curves_long.csv", d / "weekly_curves_long.csv"]
        nested = sorted(d.glob("*/long.csv"))
        curves_path = next((p for p in candidates + nested if p.exists()), candidates[0])
        return cls(
            root=d,
            curves_long_csv=curves_path,
            usdmyr_csv=d / "public" / "fx" / "usdmyr.csv",
            eurusd_csv=d / "public" / "fx" / "usdeur.csv",
            brl_monthly_csv=d / "pxm" / "BRL_monthly.csv",
            brl_daily_csv=d / "pxd" / "BRL.csv",
            dxy_monthly_csv=d / "pxm" / "DXY_monthly.csv",
            dxy_daily_csv=d / "pxd" / "DXY.csv",
            nino_txt=d / "nino_indices.txt",
            cot_history_csv=d / "public" / "cot_history.csv",
            iv_carry_csv=d / "public" / "stonex_iv_carry_2026-10-30.csv",
            pxd_dir=d / "pxd",
        )


def contract_to_months(c_series: pd.Series) -> pd.Series:
    """Convert `YYYY-MM` contract identifier strings into integer month counts."""
    parts = c_series.astype(str).str.split("-", expand=True)
    return parts[0].astype(int) * 12 + parts[1].astype(int)


def compute_cot_tradeable_availability(report_dates: pd.Series) -> pd.Series:
    """
    Map CFTC Tuesday report dates to the next Monday settlement date (`+6` days).
    CFTC publishes COT on Friday at 15:30 ET — AFTER CBOT grain/oilseed (13:15 CT / 14:15 ET)
    and ICE NY softs Friday settlements — so a Friday settlement decision only uses the
    prior week's already-published COT, with zero post-settlement lookahead.
    """
    dt = pd.to_datetime(report_dates)
    days_to_mon = (7 - dt.dt.dayofweek) % 7
    days_to_mon = np.where(days_to_mon == 0, 7, days_to_mon)
    return dt + pd.to_timedelta(days_to_mon, unit="D")


def compute_monthly_sst_availability(year: int, month: int, lag_days: int = SST_MONTHLY_RELEASE_LAG_DAYS) -> pd.Timestamp:
    """Map monthly reconstructed SST observations to `MonthEnd(0) + lag_days`."""
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag_days)


def load_macro_fx_enso_cot(paths: DataPaths) -> dict:
    """
    Load FX, multi-region NOAA ENSO SST anomalies, and CFTC Legacy Non-Commercial
    positioning with realistic point-in-time availability timestamps.
    """
    usdmyr = (
        pd.read_csv(paths.usdmyr_csv, parse_dates=["observation_date"])
        .rename(columns={"observation_date": "date", "DEXMAUS": "usdmyr"})
        .dropna()
        .sort_values("date")
    )
    eurusd = (
        pd.read_csv(paths.eurusd_csv, parse_dates=["observation_date"])
        .rename(columns={"observation_date": "date", "DEXUSEU": "eurusd"})
        .dropna()
        .sort_values("date")
    )

    # Monthly macro backfills stamped YYYY-MM-01 are shifted to MonthEnd(0) to prevent within-month lookahead
    brl_m = pd.read_csv(paths.brl_monthly_csv, parse_dates=["date"]).rename(columns={"close": "usdbrl"})
    brl_m["date"] = brl_m["date"] + pd.offsets.MonthEnd(0)
    brl_d = pd.read_csv(paths.brl_daily_csv, parse_dates=["date"])[["date", "close"]].rename(columns={"close": "usdbrl"})
    brl = (
        pd.concat([brl_m[brl_m["date"] < brl_d["date"].min()], brl_d], ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )

    dxy_m = pd.read_csv(paths.dxy_monthly_csv, parse_dates=["date"]).rename(columns={"close": "dxy"})
    dxy_m["date"] = dxy_m["date"] + pd.offsets.MonthEnd(0)
    dxy_d = pd.read_csv(paths.dxy_daily_csv, parse_dates=["date"])[["date", "close"]].rename(columns={"close": "dxy"})
    dxy = (
        pd.concat([dxy_m[dxy_m["date"] < dxy_d["date"].min()], dxy_d], ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )

    # Multi-region NOAA SST anomalies (+10-day operational release lag on historical monthly series;
    # live monitoring uses daily 5km / weekly OISST snapshots)
    nino_rows = []
    with open(paths.nino_txt, "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            toks = line.split()
            if len(toks) >= 10:
                yr, mon = int(toks[0]), int(toks[1])
                n12_a, n3_a, n4_a, n34_a = float(toks[3]), float(toks[5]), float(toks[7]), float(toks[9])
                nino_rows.append({
                    "avail_date": compute_monthly_sst_availability(yr, mon),
                    "nino12": n12_a,
                    "nino3": n3_a,
                    "nino4": n4_a,
                    "nino34": n34_a,
                    "trans_nino": n12_a - n4_a,
                })
    nino = pd.DataFrame(nino_rows).sort_values("avail_date")
    nino = (
        pd.concat([
            nino,
            pd.DataFrame([{
                "avail_date": pd.Timestamp("2026-10-29"),
                "nino12": 4.69, "nino3": 3.85, "nino4": 0.95, "nino34": 3.10, "trans_nino": 3.74,
            }]),
        ], ignore_index=True)
        .sort_values("avail_date")
        .drop_duplicates("avail_date", keep="last")
    )

    # CFTC COT: Tuesday cutoff -> Friday 15:30 ET release (after 14:15 ET CME settle) -> tradeable next Monday
    cot_raw = pd.read_csv(paths.cot_history_csv, parse_dates=["date"])
    cot_map: Dict[str, pd.DataFrame] = {}
    cot_latest_snapshot: Dict[str, dict] = {}
    for cname, grp in cot_raw.groupby("name"):
        g = grp.sort_values("date").drop_duplicates("date", keep="last").copy()
        g["avail_date"] = compute_cot_tradeable_availability(g["date"])
        g["cot_z"] = (
            (g["net_pct_oi"] - g["net_pct_oi"].rolling(156, min_periods=52).mean())
            / g["net_pct_oi"].rolling(156, min_periods=52).std()
        )
        g["cot_chg13"] = g["net_pct_oi"] - g["net_pct_oi"].shift(13)
        r_last = g.iloc[-1]
        cot_latest_snapshot[cname] = {
            "report_tuesday": str(r_last["date"].date()),
            "released_friday_1530et": str((r_last["date"] + pd.Timedelta(days=3)).date()),
            "first_post_release_session": str(r_last["avail_date"].date()),
            "net_pct_oi": round(float(r_last["net_pct_oi"]), 2),
            "cot_z": round(float(r_last["cot_z"]), 2),
        }
        cot_map[cname] = g.rename(columns={"date": "cot_report_date"})[
            ["cot_report_date", "avail_date", "net_pct_oi", "cot_z", "cot_chg13"]
        ].sort_values("avail_date")

    return {
        "usdmyr": usdmyr,
        "eurusd": eurusd,
        "brl": brl,
        "dxy": dxy,
        "nino": nino,
        "cot": cot_map,
        "cot_latest_snapshot": cot_latest_snapshot,
    }


def to_usd_per_tonne(close: pd.Series, unit: str, usdmyr: pd.Series, eurusd: pd.Series) -> pd.Series:
    """Convert native contract quotes into standardized USD/tonne (or native USD for energy)."""
    if unit == "MYR/t":
        return close / usdmyr
    if unit == "EUR/t":
        return close * eurusd
    if unit == "CAD/t":
        return close / 1.36
    if unit == "USc/lb":
        return close * 22.0462262
    if unit == "USc/kg":
        return close * 10.0
    if unit == "USD/short ton":
        return close * 1.1023113
    if unit == "USD/bu":
        return close * 36.7437  # Wheat/Soybeans 36.7437 bu/t; Corn handled explicitly at 39.36826 bu/t
    return close


def compute_contract_matched_returns(sub: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Compute exact contract-ID-matched (`YYYY-MM`) weekly log excess returns across
    tenors M1..M4, eliminating artificial continuation roll jumps when contracts expire.
    """
    sub = sub.copy()
    sub["c_months"] = contract_to_months(sub["contract"])
    piv_p = sub.pivot_table(index="date", columns="tenor", values="close", aggfunc="last").sort_index()
    piv_c = sub.pivot_table(index="date", columns="tenor", values="contract", aggfunc="last").sort_index()
    piv_m = sub.pivot_table(index="date", columns="tenor", values="c_months", aggfunc="last").sort_index()

    dc_lookup = sub.set_index(["date", "contract"])["close"].to_dict()
    dates = list(piv_p.index)

    ret_exact = {t: np.full(len(dates), np.nan) for t in (1, 2, 3, 4)}
    ret_m3_naive = np.full(len(dates), np.nan)
    matched_m3_count = 0
    total_transitions = max(len(dates) - 1, 1)

    for i in range(len(dates) - 1):
        d_nxt = dates[i + 1]
        p3_now = piv_p.iloc[i].get(3, np.nan)
        p3_nxt_naive = piv_p.iloc[i + 1].get(3, np.nan)
        if pd.notna(p3_now) and pd.notna(p3_nxt_naive) and p3_now > 0 and p3_nxt_naive > 0:
            ret_m3_naive[i + 1] = math.log(p3_nxt_naive / p3_now)

        for t_idx in (1, 2, 3, 4):
            ct = piv_c.iloc[i].get(t_idx, np.nan)
            pt_now = piv_p.iloc[i].get(t_idx, np.nan)
            if pd.notna(ct) and pd.notna(pt_now) and pt_now > 0:
                pt_nxt = dc_lookup.get((d_nxt, ct), np.nan)
                if pd.notna(pt_nxt) and pt_nxt > 0:
                    ret_exact[t_idx][i + 1] = math.log(pt_nxt / pt_now)
                    if t_idx == 3:
                        matched_m3_count += 1

    df = pd.DataFrame(index=piv_p.index)
    df.index.name = "date"
    for t in (1, 2, 3, 4, 5):
        df[f"p{t}"] = piv_p.get(t, np.nan)
        df[f"c{t}"] = piv_c.get(t, np.nan)
        df[f"m{t}"] = piv_m.get(t, np.nan)

    for t_idx in (1, 2, 3, 4):
        df[f"ret_m{t_idx}"] = ret_exact[t_idx]
    df["ret_m3_naive"] = ret_m3_naive
    return df, matched_m3_count, total_transitions


def build_weekly_market_panel(long_df: pd.DataFrame, ext: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct the weekly Friday-aligned cross-market curve panel with exact
    contract-matched excess returns, calendar-month-spaced carry, backward-merged FX,
    and Friday-available CFTC COT positioning.
    """
    clean_long = long_df[~long_df["unit"].str.contains("unconfirmed", na=False)].copy()
    clean_long = clean_long[clean_long["date"] >= pd.Timestamp("1998-01-01")].copy()

    audit_rows = []
    market_frames = []

    for name, meta in UNIVERSE_META.items():
        sub = clean_long[
            (clean_long["market"] == meta["market"]) & (clean_long["exchange"] == meta["exchange"])
        ].copy()
        if sub.empty:
            continue
        unit = sub["unit"].iloc[0]

        df, matched_m3_count, total_transitions = compute_contract_matched_returns(sub)

        # Annualized carry using exact calendar-month spacing between contracts
        dm_13 = (df["m3"] - df["m1"]).replace(0, np.nan)
        dm_15 = (df["m5"] - df["m1"]).replace(0, np.nan)
        dm_35 = (df["m5"] - df["m3"]).replace(0, np.nan)
        df["carry_13"] = (12.0 / dm_13) * np.log(df["p1"] / df["p3"])
        df["carry_15"] = (12.0 / dm_15) * np.log(df["p1"] / df["p5"])
        carry_35 = (12.0 / dm_35) * np.log(df["p3"] / df["p5"])
        df["curve_fly"] = df["carry_13"] - carry_35

        df = df.reset_index()
        df["wdate"] = df["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

        # Strictly backward-merged FX onto exact market settlement `date`
        df = pd.merge_asof(df.sort_values("date"), ext["usdmyr"], on="date", direction="backward")
        df = pd.merge_asof(df.sort_values("date"), ext["eurusd"], on="date", direction="backward")

        if name == "Corn (CBOT)":
            df["p1_usd_t"] = df["p1"] * 39.36826
            df["p3_usd_t"] = df["p3"] * 39.36826
        else:
            df["p1_usd_t"] = to_usd_per_tonne(df["p1"], unit, df["usdmyr"], df["eurusd"])
            df["p3_usd_t"] = to_usd_per_tonne(df["p3"], unit, df["usdmyr"], df["eurusd"])

        df["cum_ret_m3"] = df["ret_m3"].fillna(0.0).cumsum()
        df["mom_13w"] = df["cum_ret_m3"] - df["cum_ret_m3"].shift(13)
        df["mom_26w"] = df["cum_ret_m3"] - df["cum_ret_m3"].shift(26)
        df["mom_52w"] = df["cum_ret_m3"] - df["cum_ret_m3"].shift(52)
        df["carry_chg_13w"] = df["carry_13"] - df["carry_13"].shift(13)
        df["rv_13w"] = df["ret_m3"].rolling(13, min_periods=8).std() * math.sqrt(52.0)
        df["rv_26w"] = df["ret_m3"].rolling(26, min_periods=13).std() * math.sqrt(52.0)

        lp = np.log(df["p3_usd_t"].replace(0, np.nan))
        df["val_5y_z"] = -1.0 * (lp - lp.rolling(260, min_periods=104).mean()) / lp.rolling(260, min_periods=104).std()

        # Strictly forward 13w & 26w cumulative contract-matched excess returns
        df["fwd_13w"] = df["cum_ret_m3"].shift(-13) - df["cum_ret_m3"]
        df["fwd_26w"] = df["cum_ret_m3"].shift(-26) - df["cum_ret_m3"]
        df["fwd_13w_end_date"] = df["date"].shift(-13)
        df.loc[df.index > len(df) - 14, ["fwd_13w", "fwd_13w_end_date"]] = [np.nan, pd.NaT]
        df.loc[df.index > len(df) - 27, "fwd_26w"] = np.nan

        cot_key = meta["cot"]
        if cot_key and cot_key in ext["cot"]:
            df = pd.merge_asof(
                df.sort_values("date"),
                ext["cot"][cot_key],
                left_on="date",
                right_on="avail_date",
                direction="backward",
            )
        else:
            df["net_pct_oi"] = np.nan
            df["cot_z"] = 0.0
            df["cot_chg13"] = 0.0

        df["cot_z"] = df["cot_z"].fillna(0.0)
        df["cot_chg13"] = df["cot_chg13"].fillna(0.0)
        df["cot_squeeze"] = -1.0 * np.clip(df["cot_z"], -3.0, 3.0)

        df["name"] = name
        df["sector"] = meta["sector"]
        df["unit"] = unit

        s2000 = df[df["date"] >= pd.Timestamp("2000-01-01")]
        ann_exact = s2000["ret_m3"].mean() * 52.0
        ann_naive = s2000["ret_m3_naive"].mean() * 52.0
        audit_rows.append({
            "name": name,
            "sector": meta["sector"],
            "start_date": str(df["date"].min().date()),
            "end_date": str(df["date"].max().date()),
            "obs_weeks": int(len(s2000)),
            "m3_match_pct": round(100.0 * matched_m3_count / total_transitions, 2),
            "ann_excess_ret_exact_pct": round(100.0 * ann_exact, 2),
            "ann_cont_ret_naive_pct": round(100.0 * ann_naive, 2),
            "roll_jump_distortion_pct_yr": round(100.0 * (ann_naive - ann_exact), 2),
            "avg_carry_13_pct": round(100.0 * s2000["carry_13"].mean(), 2),
        })
        market_frames.append(df)

    panel = pd.concat(market_frames, ignore_index=True)
    panel = panel.sort_values(["name", "date"]).drop_duplicates(["name", "wdate"], keep="last")
    return panel, pd.DataFrame(audit_rows)
