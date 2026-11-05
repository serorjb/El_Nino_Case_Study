# -*- coding: utf-8 -*-
"""
Target-end-date purged (`h = 13w`) expanding-window walk-forward engine
(2006–2026), 5-stage factor ablation, and live cross-sectional ranking.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .config import REBALANCE_STEP_WEEKS, WALKFORWARD_UNIVERSE


BASE_FEATURES: List[str] = [
    "carry_13", "carry_15", "curve_fly", "carry_chg_13w",
    "rv_spread_z", "val_5y_z", "mom_13w", "mom_26w",
    "cot_squeeze", "cot_chg13",
]

DYNAMIC_PRIORS: List[str] = ["enso_kernel_pred", "macro_fx_pred", "enso_x_tight_curve"]


def newey_west_tstat(r: np.ndarray, max_lag: int = 3) -> float:
    """Newey-West HAC t-statistic testing mean return != 0."""
    x = np.asarray(r, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 12:
        return 0.0
    mu = float(np.mean(x))
    e = x - mu
    var_nw = float(np.dot(e, e) / n)
    for lag in range(1, min(max_lag + 1, n - 1)):
        w = 1.0 - lag / (max_lag + 1.0)
        cov = float(np.dot(e[lag:], e[:-lag]) / n)
        var_nw += 2.0 * w * cov
    se = math.sqrt(max(var_nw, 1e-12) / n)
    return mu / se if se > 0 else 0.0


def select_purged_training_rows(df: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    """
    Enforce strict forward-target purge: every training row's 13-week target end date
    must be on or before `decision_date` (`fwd_13w_end_date <= decision_date`).
    """
    return df[(df["fwd_13w_end_date"] <= decision_date) & df["fwd_13w"].notna()]


def _fit_expanding_priors(train_raw: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Estimate regularized out-of-sample ENSO and Macro-FX Ridge sensitivities per market."""
    enso_betas: Dict[str, np.ndarray] = {}
    macro_betas: Dict[str, np.ndarray] = {}
    for m_name, grp in train_raw.groupby("name"):
        gc = grp.dropna(subset=["fwd_13w", "nino34", "nino34_lag26", "nino34_lag52", "trans_nino"])
        if len(gc) >= 40:
            Xc = np.column_stack([
                np.ones(len(gc)), gc["nino34"].values, gc["nino34_lag26"].values,
                gc["nino34_lag52"].values, gc["trans_nino"].values,
            ])
            enso_betas[m_name] = np.linalg.solve(
                Xc.T @ Xc + np.diag([0.0, 25.0, 25.0, 25.0, 35.0]), Xc.T @ gc["fwd_13w"].values
            )
        else:
            enso_betas[m_name] = np.zeros(5)

        gm = grp.dropna(subset=["fwd_13w", "dxy_chg13", "brl_chg13", "crack_z"])
        if len(gm) >= 40:
            Xm = np.column_stack([
                np.ones(len(gm)), gm["dxy_chg13"].values, gm["brl_chg13"].values, gm["crack_z"].values
            ])
            macro_betas[m_name] = np.linalg.solve(
                Xm.T @ Xm + np.diag([0.0, 20.0, 20.0, 20.0]), Xm.T @ gm["fwd_13w"].values
            )
        else:
            macro_betas[m_name] = np.zeros(4)
    return enso_betas, macro_betas


def _attach_priors(
    df_part: pd.DataFrame, enso_betas: Dict[str, np.ndarray], macro_betas: Dict[str, np.ndarray]
) -> pd.DataFrame:
    d = df_part.copy()
    bc_mat = np.vstack([enso_betas.get(nm, np.zeros(5)) for nm in d["name"].values])
    xc_mat = np.column_stack([
        np.ones(len(d)),
        np.nan_to_num(d["nino34"].values, nan=0.0),
        np.nan_to_num(d["nino34_lag26"].values, nan=0.0),
        np.nan_to_num(d["nino34_lag52"].values, nan=0.0),
        np.nan_to_num(d["trans_nino"].values, nan=0.0),
    ])
    bm_mat = np.vstack([macro_betas.get(nm, np.zeros(4)) for nm in d["name"].values])
    xm_mat = np.column_stack([
        np.ones(len(d)),
        np.nan_to_num(d["dxy_chg13"].values, nan=0.0),
        np.nan_to_num(d["brl_chg13"].values, nan=0.0),
        np.nan_to_num(d["crack_z"].values, nan=0.0),
    ])
    d["enso_kernel_pred"] = np.sum(bc_mat * xc_mat, axis=1)
    d["macro_fx_pred"] = np.sum(bm_mat * xm_mat, axis=1)
    d["enso_x_tight_curve"] = d["enso_kernel_pred"] * np.maximum(d["carry_13"].fillna(0.0).values, -0.05)
    return d


def compute_live_signals(panel: pd.DataFrame, latest_wdate: pd.Timestamp) -> pd.DataFrame:
    """Compute strictly purged full-history cross-sectional rank forecast on the latest date."""
    test_cur = panel[panel["wdate"] == latest_wdate].copy()
    decision_date = test_cur["date"].min()
    train_raw = select_purged_training_rows(panel[panel["wdate"] >= pd.Timestamp("2000-01-01")], decision_date).copy()

    enso_betas, macro_betas = _fit_expanding_priors(train_raw)
    train_sub = _attach_priors(train_raw[train_raw["wdate"].isin(sorted(train_raw["wdate"].unique())[::2])], enso_betas, macro_betas)
    test_cur = _attach_priors(test_cur, enso_betas, macro_betas)

    all_feats = BASE_FEATURES + DYNAMIC_PRIORS
    for c in all_feats:
        train_sub[c] = train_sub[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        test_cur[c] = test_cur[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        g_mu = train_sub.groupby("wdate")[c].transform("mean")
        g_sd = train_sub.groupby("wdate")[c].transform("std").fillna(0.0) + 1e-8
        train_sub[c + "_z"] = (train_sub[c] - g_mu) / g_sd
        test_cur[c + "_z"] = (test_cur[c] - test_cur[c].mean()) / (test_cur[c].std() + 1e-8)

    train_sub["y_sharpe"] = (
        train_sub["fwd_13w"] / (train_sub["rv_26w"].fillna(0.22).clip(lower=0.10) * math.sqrt(13.0 / 52.0))
    ).clip(-3.5, 3.5)
    train_sub = train_sub.dropna(subset=["y_sharpe"]).copy()

    fcols = [c + "_z" for c in all_feats]
    ridge = Ridge(alpha=120.0, fit_intercept=False).fit(train_sub[fcols].values, train_sub["y_sharpe"].values)
    lgbm = lgb.LGBMRegressor(
        n_estimators=45, max_depth=2, num_leaves=4, learning_rate=0.05,
        min_child_samples=80, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=5.0, reg_lambda=15.0, random_state=42, verbose=-1, n_jobs=1,
    ).fit(train_sub[fcols].values, train_sub["y_sharpe"].values)

    raw_pred = 0.75 * ridge.predict(test_cur[fcols].values) + 0.25 * lgbm.predict(test_cur[fcols].values)
    test_cur["alpha_z"] = (raw_pred - np.mean(raw_pred)) / (np.std(raw_pred) + 1e-8)
    test_cur["exp_ret_6m_pct"] = np.round(
        100.0 * test_cur["alpha_z"] * test_cur["rv_26w"].fillna(0.22).clip(0.14, 0.50) * 0.28, 2
    )
    return test_cur


def run_purged_walkforward(panel: pd.DataFrame) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Run 4-week rebalanced (2006–2026) expanding-window walk-forward diagnostic across
    20 markets with continuous history, enforcing `fwd_13w_end_date <= decision_date`.
    """
    wf = panel[panel["name"].isin(WALKFORWARD_UNIVERSE) & (panel["wdate"] >= pd.Timestamp("2000-01-01"))].copy()
    wf = wf.sort_values(["wdate", "name"]).reset_index(drop=True)

    all_wdates = sorted(wf["wdate"].unique())
    start_idx = next(i for i, d in enumerate(all_wdates) if d >= pd.Timestamp("2006-01-01"))
    rebal_dates = all_wdates[start_idx::REBALANCE_STEP_WEEKS]

    ret_piv = wf.pivot(index="wdate", columns="name", values="ret_m3").sort_index()

    ablation_records: Dict[str, List[dict]] = {
        "Benchmark_EW": [],
        "M1_ENSO_Only": [],
        "M2_Curve_Val_Mom": [],
        "M3_Curve_RV_COT": [],
        "M4_Full_MultiFactor": [],
        "M5_BL_VolTarget_Net8bps": [],
        "M5_BL_Gross0bps": [],
    }
    ic_list: List[dict] = []
    prev_bl_w: Dict[str, float] = {}

    for c in BASE_FEATURES:
        wf[c] = wf[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        wf[c + "_z"] = wf.groupby("wdate")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
    wf["y_sharpe"] = (
        wf["fwd_13w"] / (wf["rv_26w"].fillna(0.22).clip(lower=0.10) * math.sqrt(13.0 / 52.0))
    ).clip(-3.5, 3.5)

    for r_i, t_date in enumerate(rebal_dates):
        test_raw = wf[wf["wdate"] == t_date]
        if len(test_raw) < 12:
            continue
        decision_date = test_raw["date"].min()
        train_raw = select_purged_training_rows(wf, decision_date)
        train_raw = train_raw[train_raw["y_sharpe"].notna()]
        if len(train_raw) < 600:
            continue

        enso_betas, macro_betas = _fit_expanding_priors(train_raw)
        train_sub = _attach_priors(train_raw[train_raw["wdate"].isin(sorted(train_raw["wdate"].unique())[::2])], enso_betas, macro_betas)
        test_cur = _attach_priors(test_raw, enso_betas, macro_betas)

        for c in DYNAMIC_PRIORS:
            g_mu = train_sub.groupby("wdate")[c].transform("mean")
            g_sd = train_sub.groupby("wdate")[c].transform("std").fillna(0.0) + 1e-8
            train_sub[c + "_z"] = (train_sub[c] - g_mu) / g_sd
            s_test = test_cur[c]
            test_cur[c + "_z"] = (s_test - s_test.mean()) / (s_test.std() + 1e-8)

        all_feats = BASE_FEATURES + DYNAMIC_PRIORS
        cols_m1 = ["enso_kernel_pred_z"]
        cols_m2 = ["carry_13_z", "carry_15_z", "val_5y_z", "mom_13w_z", "mom_26w_z"]
        cols_m3 = cols_m2 + ["carry_chg_13w_z", "curve_fly_z", "rv_spread_z", "cot_squeeze_z", "cot_chg13_z"]
        cols_m4 = [c + "_z" for c in all_feats]

        def fit_predict_ridge(fcols: List[str], alpha: float) -> np.ndarray:
            mdl = Ridge(alpha=alpha, fit_intercept=False)
            mdl.fit(train_sub[fcols].values, train_sub["y_sharpe"].values)
            return mdl.predict(test_cur[fcols].values)

        s_m1 = fit_predict_ridge(cols_m1, alpha=50.0)
        s_m2 = fit_predict_ridge(cols_m2, alpha=80.0)
        s_m3 = fit_predict_ridge(cols_m3, alpha=100.0)
        s_m4_ridge = fit_predict_ridge(cols_m4, alpha=120.0)

        lgbm = lgb.LGBMRegressor(
            n_estimators=45, max_depth=2, num_leaves=4, learning_rate=0.05,
            min_child_samples=80, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=5.0, reg_lambda=15.0, random_state=42, verbose=-1, n_jobs=1,
        )
        lgbm.fit(train_sub[cols_m4].values, train_sub["y_sharpe"].values)
        s_m4 = 0.75 * s_m4_ridge + 0.25 * lgbm.predict(test_cur[cols_m4].values)

        test_cur["score_m1"] = s_m1
        test_cur["score_m2"] = s_m2
        test_cur["score_m3"] = s_m3
        test_cur["score_m4"] = s_m4
        test_cur["exp_ann_ret"] = test_cur["score_m4"] * test_cur["rv_26w"].fillna(0.22).clip(0.12, 0.55) * 0.35

        if r_i >= len(rebal_dates) - 1:
            continue

        nxt_date = rebal_dates[r_i + 1]
        hold_slice = ret_piv[(ret_piv.index > t_date) & (ret_piv.index <= nxt_date)]
        if hold_slice.empty:
            continue
        names = list(test_cur["name"].values)
        log_4w = hold_slice.reindex(columns=names).sum(axis=0, min_count=1)
        real_4w = np.expm1(log_4w.fillna(0.0).values)

        if test_cur["fwd_13w"].notna().sum() >= 10:
            valid = test_cur.dropna(subset=["fwd_13w"])
            ic_m1, _ = spearmanr(valid["score_m1"], valid["fwd_13w"])
            ic_m3, _ = spearmanr(valid["score_m3"], valid["fwd_13w"])
            ic_m4, _ = spearmanr(valid["score_m4"], valid["fwd_13w"])
            ic_list.append({"date": t_date, "ic_m1": ic_m1, "ic_m3": ic_m3, "ic_m4": ic_m4})

        # Trailing 104-week covariance matrix with diagonal shrinkage (0.45)
        hist_ret = ret_piv[ret_piv.index <= t_date].reindex(columns=names).tail(104).fillna(0.0)
        cov_ann = hist_ret.cov().values * 52.0
        diag_cov = np.diag(np.maximum(np.diag(cov_ann), 0.015))
        S_shrunk = 0.55 * cov_ann + 0.45 * diag_cov

        def vol_target_ls_weights(raw_scores: np.ndarray, target_vol: float = 0.08) -> np.ndarray:
            z = raw_scores - np.mean(raw_scores)
            inv_vol = 1.0 / np.sqrt(np.diag(S_shrunk))
            w = z * inv_vol
            if np.sum(np.abs(w)) > 1e-8:
                w = w / np.sum(np.abs(w))
            port_v = math.sqrt(max(float(w @ S_shrunk @ w), 1e-8))
            return np.clip(w * (target_vol / port_v), -0.22, 0.22)

        w_ew = np.ones(len(names)) / len(names)
        p_ew = math.sqrt(max(float(w_ew @ S_shrunk @ w_ew), 1e-8))
        w_ew = w_ew * (0.08 / p_ew)

        w_m1 = vol_target_ls_weights(s_m1)
        w_m2 = vol_target_ls_weights(s_m2)
        w_m3 = vol_target_ls_weights(s_m3)
        w_m4 = vol_target_ls_weights(s_m4)

        # Black-Litterman posterior sizing + turnover smoothing
        tau = 0.05
        Q = test_cur["exp_ann_ret"].values - np.mean(test_cur["exp_ann_ret"].values)
        Omega_inv = np.diag(1.0 / np.maximum(np.diag(S_shrunk) * 0.25, 1e-5))
        tauS_inv = np.linalg.inv(tau * S_shrunk)
        mu_bl = np.linalg.solve(tauS_inv + Omega_inv, Omega_inv @ Q)
        w_bl_raw = np.linalg.solve(3.0 * S_shrunk, mu_bl)
        w_bl_raw = w_bl_raw - np.mean(w_bl_raw)
        pv_bl = math.sqrt(max(float(w_bl_raw @ S_shrunk @ w_bl_raw), 1e-8))
        w_bl_target = np.clip(w_bl_raw * (0.08 / pv_bl), -0.22, 0.22)

        prev_vec = np.array([prev_bl_w.get(n, 0.0) for n in names])
        w_bl = 0.60 * w_bl_target + 0.40 * prev_vec
        pv_bl2 = math.sqrt(max(float(w_bl @ S_shrunk @ w_bl), 1e-8))
        if pv_bl2 > 1e-6:
            w_bl = np.clip(w_bl * (0.08 / pv_bl2), -0.22, 0.22)

        turnover = float(np.sum(np.abs(w_bl - prev_vec)))
        prev_bl_w = {n: float(wv) for n, wv in zip(names, w_bl)}

        ablation_records["Benchmark_EW"].append({"date": nxt_date, "ret": float(np.dot(w_ew, real_4w))})
        ablation_records["M1_ENSO_Only"].append({"date": nxt_date, "ret": float(np.dot(w_m1, real_4w))})
        ablation_records["M2_Curve_Val_Mom"].append({"date": nxt_date, "ret": float(np.dot(w_m2, real_4w))})
        ablation_records["M3_Curve_RV_COT"].append({"date": nxt_date, "ret": float(np.dot(w_m3, real_4w))})
        ablation_records["M4_Full_MultiFactor"].append({"date": nxt_date, "ret": float(np.dot(w_m4, real_4w))})
        raw_bl_ret = float(np.dot(w_bl, real_4w))
        ablation_records["M5_BL_Gross0bps"].append({"date": nxt_date, "ret": raw_bl_ret, "turnover": turnover})
        ablation_records["M5_BL_VolTarget_Net8bps"].append({
            "date": nxt_date, "ret": raw_bl_ret - 0.0008 * turnover, "turnover": turnover
        })

    live_df = compute_live_signals(panel, wf["wdate"].max())

    periods_per_yr = 52.0 / float(REBALANCE_STEP_WEEKS)
    bt_summary: dict = {}
    bt_curves = pd.DataFrame()
    for k, rows in ablation_records.items():
        df_r = pd.DataFrame(rows).set_index("date")
        r = df_r["ret"]
        ann_ret = float(r.mean() * periods_per_yr)
        ann_vol = float(r.std(ddof=1) * math.sqrt(periods_per_yr))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        eq = (1.0 + r).cumprod()
        mdd = float((eq / eq.cummax() - 1.0).min())
        bt_summary[k] = {
            "ann_excess_return_pct": round(100.0 * ann_ret, 2),
            "ann_vol_pct": round(100.0 * ann_vol, 2),
            "sharpe": round(sharpe, 2),
            "newey_west_tstat": round(newey_west_tstat(r.values, max_lag=3), 2),
            "max_drawdown_pct": round(100.0 * mdd, 2),
            "hit_rate_pct": round(100.0 * float((r > 0).mean()), 1),
            "n_periods": int(len(r)),
            "avg_4w_turnover": round(float(df_r["turnover"].mean()), 3) if "turnover" in df_r else None,
        }
        bt_curves[k] = eq

    ic_df = pd.DataFrame(ic_list)
    bt_summary["rank_ic"] = {
        "m1_enso_mean_ic": round(float(ic_df["ic_m1"].mean()), 3),
        "m3_curve_rv_cot_mean_ic": round(float(ic_df["ic_m3"].mean()), 3),
        "m4_full_mean_ic": round(float(ic_df["ic_m4"].mean()), 3),
        "m4_ic_tstat": round(newey_west_tstat(ic_df["ic_m4"].values, max_lag=3), 2),
    }

    return bt_summary, bt_curves, live_df
