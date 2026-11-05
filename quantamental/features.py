# -*- coding: utf-8 -*-
"""
Cross-commodity physical processing/substitution margins and lagged ENSO / FX
state features aligned strictly to each market's exact observation `date`.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def add_cross_commodity_and_macro_features(
    panel: pd.DataFrame, long_df: pd.DataFrame, ext: dict
) -> Tuple[pd.DataFrame, dict]:
    """
    Compute physical processing/substitution spreads and lagged climate/FX features
    strictly on each row's exact settlement `date` (never peeking across week-end `wdate`).
    """
    cal_idx = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    piv_usd_d = (
        panel.pivot_table(index="date", columns="name", values="p3_usd_t", aggfunc="last")
        .reindex(cal_idx)
        .ffill(limit=10)
    )
    piv_p3_d = (
        panel.pivot_table(index="date", columns="name", values="p3", aggfunc="last")
        .reindex(cal_idx)
        .ffill(limit=10)
    )

    def rz_cal(s: pd.Series, win: int = 1820, minp: int = 546) -> pd.Series:
        return (s - s.rolling(win, min_periods=minp).mean()) / s.rolling(win, min_periods=minp).std()

    # 1. Inter-Vegoil Spread: BOPO = Soyoil - Palm (USD/t); POBO = Palm - Soyoil (USD/t)
    bopo = piv_usd_d["Soybean Oil"] - piv_usd_d["Palm Oil"]
    bopo_z = rz_cal(bopo)

    # 2. Biodiesel Economics: POGO = Palm Oil - ICE Gasoil (USD/t)
    pogo = piv_usd_d["Palm Oil"] - piv_usd_d["ICE Gasoil"]
    pogo_z = rz_cal(pogo)

    # 3. CBOT Board Crush & Oil Share (%)
    crush = piv_p3_d["Soybean Meal"] * 0.022 + piv_p3_d["Soybean Oil"] * 0.11 - piv_p3_d["Soybeans"]
    oil_val = piv_p3_d["Soybean Oil"] * 0.11
    meal_val = piv_p3_d["Soybean Meal"] * 0.022
    oil_share = oil_val / (oil_val + meal_val)
    crush_z = rz_cal(crush)
    oil_share_z = rz_cal(oil_share)

    # 4. Contract-Matched White (#5) vs Raw (#11) Sugar Refining Margin (USD/t)
    sug_sub = long_df[
        long_df["market"].isin(["White sugar No. 5", "Raw sugar No. 11"])
        & (long_df["date"] >= pd.Timestamp("1998-01-01"))
    ].copy()
    wh = sug_sub[sug_sub["market"] == "White sugar No. 5"][["date", "contract", "close", "tenor"]].rename(
        columns={"close": "wh_usd", "tenor": "wh_tenor"}
    )
    rw = sug_sub[sug_sub["market"] == "Raw sugar No. 11"][["date", "contract", "close", "tenor"]].rename(
        columns={"close": "rw_c", "tenor": "rw_tenor"}
    )
    wr_m = pd.merge(wh, rw, on=["date", "contract"], how="inner")
    wr_m["wr_spread"] = wr_m["wh_usd"] - wr_m["rw_c"] * 22.0462262
    wr_front = wr_m.sort_values(["date", "wh_tenor"]).drop_duplicates("date", keep="first").set_index("date")["wr_spread"]
    wr_z = rz_cal(wr_front.reindex(cal_idx).ffill(limit=14))

    # Live Oct 29, 2026 contract-matched spread snapshot
    oct29 = long_df[long_df["date"] == pd.Timestamp("2026-10-29")]
    fx_oct = float(ext["usdmyr"][ext["usdmyr"]["date"] <= pd.Timestamp("2026-10-29")].iloc[-1]["usdmyr"])
    fcpo_oct = oct29[oct29["market"] == "Crude palm oil"].set_index("contract")["close"]
    zl_oct = oct29[oct29["market"] == "Soybean oil"].set_index("contract")["close"]
    lgo_oct = oct29[oct29["market"] == "Low sulphur gasoil"].set_index("contract")["close"]

    matched_vegoil = {}
    for cm in ("2026-12", "2027-01", "2027-03"):
        if cm in fcpo_oct and cm in zl_oct:
            p_usd = float(fcpo_oct[cm]) / fx_oct
            z_usd = float(zl_oct[cm]) * 22.0462262
            g_usd = float(lgo_oct[cm]) if cm in lgo_oct else np.nan
            matched_vegoil[cm] = {
                "fcpo_myr_t": float(fcpo_oct[cm]),
                "fcpo_usd_t": round(p_usd, 2),
                "zl_usc_lb": float(zl_oct[cm]),
                "zl_usd_t": round(z_usd, 2),
                "pobo_usd_t": round(p_usd - z_usd, 2),
                "pogo_usd_t": round(p_usd - g_usd, 2) if pd.notna(g_usd) else None,
            }

    sb_oct29 = oct29[oct29["market"] == "Raw sugar No. 11"].set_index("contract")["close"]
    kc_oct29 = oct29[oct29["market"] == "Coffee arabica"].set_index("contract")["close"]
    sugar_mar_oct_c = (
        round(float(sb_oct29["2027-03"] - sb_oct29["2027-10"]), 2)
        if ("2027-03" in sb_oct29 and "2027-10" in sb_oct29)
        else np.nan
    )
    arabica_may_sep_c = (
        round(float(kc_oct29["2027-05"] - kc_oct29["2027-09"]), 2)
        if ("2027-05" in kc_oct29 and "2027-09" in kc_oct29)
        else np.nan
    )

    # 5. Wheat Protein (HRW - SRW) & Feed (SRW / Corn)
    kc_srw = piv_p3_d["Wheat HRW"] - piv_p3_d["Wheat SRW"]
    kc_srw_z = rz_cal(kc_srw)
    wheat_corn_ratio = np.log(piv_p3_d["Wheat SRW"] / piv_p3_d["Corn (CBOT)"])
    wheat_corn_z = rz_cal(wheat_corn_ratio)

    # 6. Coffee Arbitrage (Arabica c/lb - Robusta converted to c/lb)
    arab_rob = piv_p3_d["Arabica Coffee"] - (piv_p3_d["Robusta Coffee"] / 22.0462262)
    arab_rob_z = rz_cal(arab_rob)

    # 7. ICE Gasoil Crack Spread (USD/bbl)
    gasoil_crack = (piv_p3_d["ICE Gasoil"] / 7.45) - piv_p3_d["Brent Crude"]
    crack_z = rz_cal(gasoil_crack)

    # Cross-commodity relative-value cheapness composite (`rv_spread_z` > 0 = cheap vs processing/substitution peer)
    rv_df = pd.DataFrame(0.0, index=cal_idx, columns=piv_usd_d.columns)
    rv_df["Palm Oil"] = 0.65 * bopo_z.fillna(0.0) - 0.35 * pogo_z.fillna(0.0)
    rv_df["Soybean Oil"] = -0.60 * bopo_z.fillna(0.0) - 0.40 * oil_share_z.fillna(0.0)
    rv_df["Soybeans"] = 0.60 * crush_z.fillna(0.0)
    rv_df["Soybean Meal"] = 0.40 * oil_share_z.fillna(0.0) - 0.30 * crush_z.fillna(0.0)
    rv_df["Canola"] = 0.40 * bopo_z.fillna(0.0)
    rv_df["Rapeseed"] = 0.35 * bopo_z.fillna(0.0) - 0.25 * pogo_z.fillna(0.0)
    rv_df["Wheat SRW"] = 0.50 * kc_srw_z.fillna(0.0) - 0.50 * wheat_corn_z.fillna(0.0)
    rv_df["Wheat HRW"] = -0.50 * kc_srw_z.fillna(0.0) - 0.30 * wheat_corn_z.fillna(0.0)
    rv_df["Milling Wheat"] = -0.40 * wheat_corn_z.fillna(0.0)
    rv_df["Corn (CBOT)"] = 0.50 * wheat_corn_z.fillna(0.0)
    rv_df["Arabica Coffee"] = -0.40 * arab_rob_z.fillna(0.0)
    rv_df["Robusta Coffee"] = 0.40 * arab_rob_z.fillna(0.0)
    rv_df["Raw Sugar #11"] = 0.55 * wr_z.fillna(0.0)
    rv_df["White Sugar #5"] = -0.55 * wr_z.fillna(0.0)
    rv_df["Brent Crude"] = 0.40 * crack_z.fillna(0.0)
    rv_df["ICE Gasoil"] = -0.40 * crack_z.fillna(0.0)

    rv_long = rv_df.loc[sorted(panel["date"].unique())].stack().reset_index()
    rv_long.columns = ["date", "name", "rv_spread_z"]
    panel = pd.merge(panel, rv_long, on=["date", "name"], how="left")
    panel["rv_spread_z"] = panel["rv_spread_z"].fillna(0.0)

    # Backward-merged Macro FX & ENSO state onto exact settlement `date`
    macro_d = pd.DataFrame({"date": cal_idx})
    macro_d = pd.merge_asof(macro_d, ext["dxy"].sort_values("date"), on="date", direction="backward")
    macro_d = pd.merge_asof(macro_d, ext["brl"].sort_values("date"), on="date", direction="backward")
    macro_d = pd.merge_asof(macro_d, ext["usdmyr"].sort_values("date"), on="date", direction="backward")
    macro_d = pd.merge_asof(
        macro_d, ext["nino"].sort_values("avail_date"), left_on="date", right_on="avail_date", direction="backward"
    )
    macro_d["dxy_chg13"] = np.log(macro_d["dxy"] / macro_d["dxy"].shift(91)).fillna(0.0)
    macro_d["brl_chg13"] = np.log(macro_d["usdbrl"] / macro_d["usdbrl"].shift(91)).fillna(0.0)
    macro_d["myr_chg13"] = np.log(macro_d["usdmyr"] / macro_d["usdmyr"].shift(91)).fillna(0.0)
    macro_d["nino34_lag26"] = macro_d["nino34"].shift(182)
    macro_d["nino34_lag52"] = macro_d["nino34"].shift(364)
    macro_d["nino34_chg13"] = macro_d["nino34"] - macro_d["nino34"].shift(91)
    macro_d["crack_z"] = crack_z.reindex(cal_idx).fillna(0.0).values

    panel = pd.merge(
        panel,
        macro_d[[
            "date", "nino34", "trans_nino", "nino34_lag26", "nino34_lag52", "nino34_chg13",
            "dxy_chg13", "brl_chg13", "myr_chg13", "crack_z", "usdbrl", "dxy",
        ]],
        on="date",
        how="left",
    )

    d_oct29 = pd.Timestamp("2026-10-29")
    spread_snapshot = {
        "matched_vegoil_oct29": matched_vegoil,
        "sugar_mar27_minus_oct27_c_lb": sugar_mar_oct_c,
        "arabica_may27_minus_sep27_c_lb": arabica_may_sep_c,
        "pobo_m3_oct29_usd_t": round(float(-bopo.loc[d_oct29]), 2),
        "pobo_m3_z": round(float(-bopo_z.loc[d_oct29]), 2),
        "pogo_m3_oct29_usd_t": round(float(pogo.loc[d_oct29]), 2),
        "cbot_oil_share_pct": round(100.0 * float(oil_share.loc[d_oct29]), 2),
    }
    return panel, spread_snapshot
