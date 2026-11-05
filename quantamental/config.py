# -*- coding: utf-8 -*-
"""
Universe configuration, publication-lag rules, physical S&D overlays, and
relative volatility-mandate parameters.
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# 1. TRADABLE COMMODITY UNIVERSE (21 LIVE MARKETS; 20 IN 2006-2026 BACKTEST)
# ---------------------------------------------------------------------------
UNIVERSE_META: Dict[str, dict] = {
    "Palm Oil": {
        "market": "Crude palm oil", "exchange": "Bursa Malaysia", "sector": "Vegoils",
        "pxd": None, "cot": None, "lot_size": "25 t", "lot_tonnes": 25.0,
    },
    "Soybean Oil": {
        "market": "Soybean oil", "exchange": "CBOT", "sector": "Vegoils",
        "pxd": "ZL.csv", "cot": "Soyoil", "lot_size": "60,000 lb", "lot_tonnes": 27.2155,
    },
    "Soybeans": {
        "market": "Soybeans", "exchange": "CBOT", "sector": "Oilseeds",
        "pxd": "ZS.csv", "cot": None, "lot_size": "5,000 bu", "lot_tonnes": 136.078,
    },
    "Soybean Meal": {
        "market": "Soybean meal", "exchange": "CBOT", "sector": "Oilseeds",
        "pxd": "ZM.csv", "cot": "Meal", "lot_size": "100 st", "lot_tonnes": 90.7185,
    },
    "Canola": {
        "market": "Canola", "exchange": "ICE US", "sector": "Oilseeds",
        "pxd": None, "cot": None, "lot_size": "20 t", "lot_tonnes": 20.0,
    },
    "Rapeseed": {
        "market": "Rapeseed", "exchange": "Euronext", "sector": "Oilseeds",
        "pxd": None, "cot": None, "lot_size": "50 t", "lot_tonnes": 50.0,
    },
    "Corn (CBOT)": {
        "market": "Corn", "exchange": "CBOT", "sector": "Grains",
        "pxd": "ZC.csv", "cot": "Corn", "lot_size": "5,000 bu", "lot_tonnes": 127.006,
    },
    "Corn (Matif)": {
        "market": "Corn", "exchange": "Euronext", "sector": "Grains",
        "pxd": None, "cot": None, "lot_size": "50 t", "lot_tonnes": 50.0,
    },
    "Wheat SRW": {
        "market": "Wheat SRW (Chicago)", "exchange": "CBOT", "sector": "Grains",
        "pxd": "ZW.csv", "cot": "Wheat", "lot_size": "5,000 bu", "lot_tonnes": 136.078,
    },
    "Wheat HRW": {
        "market": "Wheat HRW (Kansas City)", "exchange": "CBOT", "sector": "Grains",
        "pxd": "KE.csv", "cot": "Wheat", "lot_size": "5,000 bu", "lot_tonnes": 136.078,
    },
    "Milling Wheat": {
        "market": "Milling wheat", "exchange": "Euronext", "sector": "Grains",
        "pxd": None, "cot": None, "lot_size": "50 t", "lot_tonnes": 50.0,
    },
    "Arabica Coffee": {
        "market": "Coffee arabica", "exchange": "ICE US", "sector": "Softs",
        "pxd": "KC.csv", "cot": "Arabica", "lot_size": "37,500 lb", "lot_tonnes": 17.0097,
    },
    "Robusta Coffee": {
        "market": "Coffee robusta", "exchange": "ICE Europe", "sector": "Softs",
        "pxd": None, "cot": None, "lot_size": "10 t", "lot_tonnes": 10.0,
    },
    "Raw Sugar #11": {
        "market": "Raw sugar No. 11", "exchange": "ICE US", "sector": "Softs",
        "pxd": "SB.csv", "cot": "Sugar", "lot_size": "112,000 lb", "lot_tonnes": 50.8023,
    },
    "White Sugar #5": {
        "market": "White sugar No. 5", "exchange": "ICE Europe", "sector": "Softs",
        "pxd": None, "cot": "Sugar", "lot_size": "50 t", "lot_tonnes": 50.0,
    },
    "Cocoa (NY)": {
        "market": "Cocoa (New York)", "exchange": "ICE US", "sector": "Softs",
        "pxd": "CC.csv", "cot": "Cocoa", "lot_size": "10 t", "lot_tonnes": 10.0,
    },
    "Cotton #2": {
        "market": "Cotton No. 2", "exchange": "ICE US", "sector": "Softs",
        "pxd": "CT.csv", "cot": "Cotton", "lot_size": "50,000 lb", "lot_tonnes": 22.6796,
    },
    "Rubber TSR20": {
        "market": "Rubber TSR20", "exchange": "SGX", "sector": "Softs",
        "pxd": None, "cot": None, "lot_size": "5 t", "lot_tonnes": 5.0,
    },
    "Brent Crude": {
        "market": "Brent crude", "exchange": "ICE Europe", "sector": "Energy",
        "pxd": "CL.csv", "cot": None, "lot_size": "1,000 bbl", "lot_tonnes": 134.0,
    },
    "ICE Gasoil": {
        "market": "Low sulphur gasoil", "exchange": "ICE Europe", "sector": "Energy",
        "pxd": None, "cot": None, "lot_size": "100 t", "lot_tonnes": 100.0,
    },
    "Henry Hub Gas": {
        "market": "Henry Hub natural gas", "exchange": "NYMEX", "sector": "Energy",
        "pxd": "NG.csv", "cot": None, "lot_size": "10,000 MMBtu", "lot_tonnes": np.nan,
    },
}

# 20 liquid ags/softs/energy curves with continuous history across 2006-2026
# (SGX Rubber TSR20 starts in Oct 2020, so it enters the live 21-market screen only)
WALKFORWARD_UNIVERSE = [k for k in UNIVERSE_META.keys() if k != "Rubber TSR20"]

# ---------------------------------------------------------------------------
# 2. REALISTIC POINT-IN-TIME RELEASE LAG & MANDATE PARAMETERS
# ---------------------------------------------------------------------------
SST_MONTHLY_RELEASE_LAG_DAYS: int = 10    # +10d operational lag after month-end (~10th of next month for ERSSTv5; live uses daily 5km / weekly OISST)
WALKFORWARD_HORIZON_WEEKS: int = 13       # 13-week forward excess-return ranking target & strict purge gap
REBALANCE_STEP_WEEKS: int = 4             # 4-week non-overlapping evaluation blocks (13 periods/year)
IV_RANK_SHORTLIST_MAX_PCT: float = 20.0   # Step-1 52-week IV Rank shortlist ceiling for long option structures

MANDATE_TARGET_VOL_PCT: float = 10.0      # Full-mandate annualized volatility ceiling (%)
DAY1_DEPLOYED_SHARE_PCT: int = 65         # 65% of vol budget deployed Day 1 -> 6.5% expected annualized volatility
CASH_RESERVE_SHARE_PCT: int = 35          # 35% unallocated risk reserve held for dated physical triggers


# ---------------------------------------------------------------------------
# 3. TRANSPARENT PHYSICAL S&D OVERLAYS (LOGGED SEPARATELY FROM MODEL PRIOR)
# ---------------------------------------------------------------------------
PHYSICAL_SD_OVERLAYS: Dict[str, Tuple[float, str]] = {
    "Palm Oil": (
        +3.5,
        "Stage-dependent SE Asia El Nino yield lag (~5-6m window) + Indo B40 pull vs ~3.75 Mt Oct survey stock peak "
        "(start at 30%, scale on MPOB < 3.30 Mt, roll Mar-27 to May/Jul-27 pre-FND, exit if Q1 > 3.85 Mt)",
    ),
    "Soybean Oil": (
        -2.5,
        "US crush +4% y/y, 50.6% Full Carry & 50.0% Oil Share + wet S. America soy planting "
        "(cap F1 short via 4% IVR 70c calls)",
    ),
    "Soybean Meal": (
        +1.5,
        "Tightest CBOT carry (15.3% Full Carry) + 50% Oil Share mean-reversion (modest 8% F4 sleeve; 0% if house crush overlaps)",
    ),
    "Arabica Coffee": (
        +3.0,
        "El Nino heat shrunk washed bean sizes (~68c diffs, tight 225k NY certs) ahead of 77.2M-bag H2 BR harvest -> +May-27/-Sep-27 calendar",
    ),
    "Robusta Coffee": (
        -2.5,
        "Two-clock split: +19.6% spot VN exports weigh on front vs Jan-Mar El Nino dry-season risk -> ZERO short Robusta; long watchlist",
    ),
    "Raw Sugar #11": (
        -1.5,
        "El Nino hemisphere split: Thai -20% H1 deficit vs 662 Mt CS Brazil H2 cane & +22.4% spec crowding -> +Mar-27/-Oct-27 calendar",
    ),
    "White Sugar #5": (
        +0.5,
        "EU beet cut + Thai -20% crop tighten H1 white exports; avoid shorting front whites",
    ),
    "Corn (CBOT)": (
        +2.0,
        "Conditional watchlist (0% active today): Brazil safrinha risk requires Jan soy harvest delays past late Feb (currently 54.3% planted)",
    ),
    "Wheat SRW": (
        +3.0,
        "Excess El Nino rain damaged S. Brazil wheat quality (7.1 Mt imports) + dry Aus risk vs -58.3k fund shorts (16% IVR May-27 call spread)",
    ),
    "Wheat HRW": (
        +2.5,
        "Tight 28.3% Full Carry vs 41.0% SRW; expressed cleanly via cheap 16% IVR SRW call spread",
    ),
    "Soybeans": (
        -0.5,
        "Two-step vol filter skip: Jan-27 (18% IVR) expires Dec 24 before Jan 10 tariff window; Mar-27 sits at 30% IVR (>20% cutoff)",
    ),
    "Cocoa (NY)": (
        -12.5,
        "Hard physical veto: +422 kt 2025/26 global surplus rebuilt stocks (+25 kt 26/27e & weak grindings) at 42.6% ATM IV overrides 2023/24 drought beta",
    ),
    "Cotton #2": (
        +1.5,
        "Dry Australian El Nino cut offsets -15.9% contango & crowded positioning -> neutral",
    ),
    "Henry Hub Gas": (
        -3.0,
        "Warm northern-tier El Nino winter + 94% IVR & extreme call skew -> vetoed",
    ),
    "Rubber TSR20": (
        -1.5,
        "Curve already 98th percentile backwardated; smallholder tapping buffer caps upside -> vetoed",
    ),
}

# Markets belonging to the active 5-sleeve Day-1 portfolio (`F1`-`F4` + `O1`);
# all other screened markets are held at `0.0%` active view via physical veto,
# two-step IVR filter rejection, or dated watchlist triggers.
ACTIVE_SLEEVE_MARKETS = frozenset({
    "Palm Oil",
    "Soybean Oil",
    "Raw Sugar #11",
    "Arabica Coffee",
    "Soybean Meal",
    "Wheat SRW",
})


def get_physical_sd_overlay(name: str) -> Tuple[float, str]:
    """Return logged discretionary S&D adjustment (6M % score units) and rationale."""
    return PHYSICAL_SD_OVERLAYS.get(
        name,
        (0.0, "Pure statistical cross-curve & macro prior (no discretionary S&D overlay)"),
    )
