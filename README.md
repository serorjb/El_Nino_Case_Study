# Quantamental Cross-Curve Commodity & ENSO Portfolio Engine

A modular, point-in-time quantitative-fundamental (**quantamental**) screening and portfolio construction framework for agricultural, softs, and energy futures curves across El Niño-Southern Oscillation (**ENSO**) regimes.

All portfolio allocations, covariance risk contributions, option costs, and stress scenarios are expressed strictly in **relative `%` terms around a `10.0%` Annualized Volatility Mandate** (zero arbitrary currency book-size assumptions; zero cash interest included in return attribution).

---

## Architecture Overview

The engine separates systematic cross-curve statistical priors from auditable physical supply-and-demand (**S&D**) governance and volatility-mandate risk construction across five sequential stages:

1. **Point-in-Time Curve & Macro Ingestion ([`quantamental/data_loader.py`](quantamental/data_loader.py))**
   - **Exact Contract-ID-Matched Excess Returns**: Matches identical `YYYY-MM` contract identifiers week-over-week (`c_t == c_{t-1}`) across tenors `M1..M4`, eliminating artificial continuation roll jumps when front contracts expire.
   - **Calendar-Spaced Carry & Butterfly**: Annualizes curve carry (`carry_13`, `carry_15`) and curvature (`curve_fly`) using exact integer contract-month spacing.
   - **Publication-Lag Hygiene**:
     - **CFTC COT**: Maps Tuesday report cutoffs (released Friday `15:30 ET`, *after* `13:15 CT / 14:15 ET` CME grain/oilseed and ICE NY softs Friday settlements) to the **next Monday session (`+6` days)** before backward `merge_asof` — ensuring Friday settlement decisions only use already-published positioning.
     - **NOAA ENSO SST**: Applies a **`+10`-day operational release lag** after month-end on historical monthly SST anomalies (`Niño 3.4`, `Niño 1+2`, `Trans-Niño`), paired with daily 5km / weekly OISST snapshots in live operation.
     - **Macro FX**: Shifts pre-2014 monthly `BRL` and `DXY` series stamped `YYYY-MM-01` to `MonthEnd(0)` prior to backward merging so intra-month closes never leak backward.

2. **Cross-Commodity Substitution & Processing Margins ([`quantamental/features.py`](quantamental/features.py))**
   - Computes standardized 5-year rolling z-scores for physical substitution and processing margins strictly on each market's exact observation `date` (no cross-day Friday `wdate` peeking):
     - **Inter-Vegoil (`POBO` / `BOPO`)**: Palm Oil vs. Soybean Oil (`USD/t`, `FX`-translated)
     - **Biodiesel Floor (`POGO`)**: Palm Oil vs. Low Sulphur Gasoil (`USD/t`)
     - **Soybean Complex**: CBOT Board Crush & Oil Share (`%`)
     - **Refining & Feed Spreads**: Contract-matched White (`#5`) vs. Raw (`#11`) Sugar; HRW-SRW Wheat protein & SRW/Corn feed ratio; Arabica-Robusta arbitrage; Gasoil crack spread.

3. **Purged Walk-Forward Diagnostic & Factor Ablation ([`quantamental/walkforward.py`](quantamental/walkforward.py))**
   - Evaluates a 4-week rebalanced (`2006–2026`, `271` non-overlapping periods) expanding-window cross-sectional model across the **`20` liquid markets** with continuous 20-year history (`SGX Rubber TSR20`, launched Oct 2020, enters the `21`-market live screen only).
   - **Strict Target-End-Date Purge**: Every training observation must satisfy `fwd_13w_end_date <= decision_date`, preventing overlapping 13-week forward returns from leaking across rebalance boundaries.
   - Combines cross-sectional `Ridge` (`75%`) + shallow `LightGBM` (`25%`) rank forecasts with `104`-week diagonal-shrunk (`0.45`) covariance and Black-Litterman turnover smoothing (`8 bps` transaction cost penalty).

4. **Physical S&D Governance & Two-Step `<= 20%` IVR Selector ([`quantamental/volatility.py`](quantamental/volatility.py))**
   - Logs three explicit columns side-by-side for every market:
     1. `prior_stat_ret_6m_pct` — pure statistical cross-curve + lagged-ENSO model prior
     2. `physical_sd_adj_pct` — structured physical S&D / crop-calendar adjustment
     3. `active_post_rule_view_6m_pct` — post-veto / post-filter executed view
   - **Two-Step Option Filter**:
     - **Step 1 (Implied Vol Shortlist)**: Verified 52-week Implied Volatility Rank (`IVR`) must be **`<= 20%`** (never inventing synthetic implied volatilities where no liquid option surface is published).
     - **Step 2 (Catalyst Expiry Match)**: Option expiry must cover the dated physical or policy catalyst window (e.g., passing `4% IVR` `Mar-27` Soyoil `70c` call caps and `16% IVR` `May-27` SRW Wheat `630/710` call spreads, while rejecting `Mar-27` Soybeans at `30% IVR > 20%` and `Jan-27` Soybeans expiring Dec 24 before the Jan 10 tariff window).

5. **Relative `10.0%` Volatility-Mandate Portfolio & Stress Engine ([`quantamental/portfolio.py`](quantamental/portfolio.py))**
   - Sizes five active sleeves (`65%` of vol budget deployed Day 1 = **`6.5%` expected annualized volatility**; Palm contributes `~45%` of active Day-1 risk) plus a **`35%` unallocated risk reserve** reserved for dated physical/weather triggers (`9.8%` post-trigger expected volatility):

| ID | Active Sleeve / Reserve | Traded Expression | Day-1 Share | Post-Trigger | Key Gate / Exit Rule |
|:--|:--|:--|--:|--:|:--|
| **F1** | Palm Oil vs. Soyoil + Call Cap | Long `Mar-27` `FCPO` / Short `Mar-27` `ZL` + `70c` `ZL` Call (`4% IVR`) | **30%** | **50%** | Scale on `MPOB < 3.30 Mt` (from `~3.75 Mt` Oct survey); exit if Q1 `> 3.85 Mt` |
| **F2** | Raw Sugar `#11` Bull Spread | Long `Mar-27` / Short `Oct-27` `#11` Sugar (`+1.06 c/lb`) | **15%** | **15%** | H1 Thai/India deficit vs. H2 Brazil cane; stop `< +0.35 c/lb` |
| **F3** | Arabica Coffee Bull Spread | Long `May-27` / Short `Sep-27` Arabica (`+5.45 c/lb`; **ZERO short Robusta**) | **11%** | **11%** | Tight `225k` NY certs ahead of H2 Brazil harvest; stop `< +0.50 c/lb` |
| **F4** | Soymeal / Soyoil Crush Value | Long `Mar-27` `ZM` / Short `Mar-27` `ZL` | **8%** | **8%** | `15.3%` vs. `50.6%` Full Carry (`0%` if house crush overlaps); stop `> 53.5%` Oil Share |
| **O1** | SRW Wheat Tail Convexity | Long `May-27` `ZW` `630/710` Call Spread (`16% IVR`) | **1%** | **1%** | Defined `0.58%` premium cap; monetize `>= 705 c/bu` |
| **Reserve** | Unallocated Risk Reserve | Dry powder held for dated physical/weather triggers | **35%** | **15%** | `0.0%` cash yield assumed in scenario returns |
| **Total** | **10.0% Annualized Vol Mandate** | **`65%` Deployed (`6.5%` Expected Vol) / `35%` Reserve** | **100%** | **100%** | **Probability-Weighted Pre-Roll Scenario Mean: `+2.5%` (`+2.54%`)** |

---

## Repository Structure

```text
quantamental_framework/
├── quantamental/
│   ├── __init__.py       # Public package exports
│   ├── config.py         # 21-market universe metadata, lag rules & physical S&D overlays
│   ├── data_loader.py    # Point-in-time FX, ENSO, post-settle COT & contract-matched curves
│   ├── features.py       # Exact-date cross-commodity substitution & processing spread builder
│   ├── walkforward.py    # Target-purged (h=13w) expanding walk-forward & 5-stage ablation
│   ├── volatility.py     # Contract-matched RV, GJR-GARCH/HAR-RV, Black-76 & <=20% IVR filter
│   └── portfolio.py      # 10.0%-vol mandate risk shares, FX sensitivity & scenario engine
├── tests/
│   └── test_engine_integrity.py  # Standalone mathematical & point-in-time unit tests (no data needed)
├── run_pipeline.py       # End-to-end CLI pipeline runner
├── requirements.txt      # Pinned Python scientific dependencies
└── .gitignore            # Excludes local market data, outputs & caches
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Standalone Unit Tests (No External Data Required)

The integrity suite verifies roll-jump-free contract matching, post-settlement CFTC COT (`Tuesday -> next Monday`) and `MonthEnd + 10d` SST lags, strict `fwd_13w_end_date <= decision_date` purging, Black-76 put-call parity, the two-step `<= 20%` IVR filter, and `100%` mandate share invariants on synthetic fixtures:

```bash
python -m unittest discover -s tests -v
```

### 3. Run Full Pipeline on Local Market Data

Market data is excluded from version control via `.gitignore`. Place your local `data/` folder outside or inside the directory following the expected layout below and run:

```bash
python run_pipeline.py --data-dir /path/to/data --output-dir ./output
```

#### Expected `--data-dir` Input Layout

- `curves/weekly_curves_long.csv` *(or `<subdir>/long.csv`)* — weekly multi-tenor (`tenor` `1..5`) futures settlements with columns: `date, market, exchange, tenor, contract, close, unit`
- `nino_indices.txt` — NOAA monthly Pacific SST anomaly table (`Niño 1+2`, `Niño 3`, `Niño 4`, `Niño 3.4`)
- `public/cot_history.csv` — CFTC Legacy Non-Commercial positioning (`date, name, net_pct_oi`)
- `public/stonex_iv_carry_2026-10-30.csv` — optional verified implied volatility rank, skew & `% Full Carry` snapshot
- `public/fx/usdmyr.csv` & `public/fx/usdeur.csv` — daily spot FX series
- `pxd/` & `pxm/` — daily (`BRL.csv`, `DXY.csv`, optional front-continuation daily series for secondary GJR-GARCH/HAR-RV diagnostics) and pre-2014 monthly (`BRL_monthly.csv`, `DXY_monthly.csv`) series

#### Generated Outputs (`--output-dir`)

- `framework_summary.json` — full methodology audit, walk-forward ablation metrics, `104`-week covariance risk decomposition, `USD/MYR` FX sensitivity, Black-76 option cost audit, `11x11` correlation matrix, and probability-weighted scenario payoffs
- `signals.csv` — 21-market cross-sectional statistical priors, physical S&D adjustments, post-rule views, volatility diagnostics, and selected structures
- `walkforward_curves.csv` — out-of-sample cumulative excess-return equity curves (`2006–2026`) across all ablation stages
- `roll_audit.csv` — contract-ID match rates and naive continuation roll-jump distortion audit across all 21 markets
