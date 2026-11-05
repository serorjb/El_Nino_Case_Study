# -*- coding: utf-8 -*-
"""
Quantamental Cross-Curve Commodity Screening & Volatility-Mandate Portfolio Engine.
"""

from .config import (
    ACTIVE_SLEEVE_MARKETS,
    CASH_RESERVE_SHARE_PCT,
    DAY1_DEPLOYED_SHARE_PCT,
    IV_RANK_SHORTLIST_MAX_PCT,
    MANDATE_TARGET_VOL_PCT,
    PHYSICAL_SD_OVERLAYS,
    UNIVERSE_META,
    WALKFORWARD_UNIVERSE,
    get_physical_sd_overlay,
)
from .data_loader import (
    DataPaths,
    build_weekly_market_panel,
    compute_contract_matched_returns,
    compute_cot_tradeable_availability,
    compute_monthly_sst_availability,
    load_macro_fx_enso_cot,
)
from .features import add_cross_commodity_and_macro_features
from .portfolio import (
    MANDATE_SLEEVE_SHARES_PCT,
    build_mandate_portfolio_and_scenarios,
    compute_cross_market_correlation,
)
from .volatility import (
    black76_option,
    evaluate_two_step_option_filter,
    forecast_volatility_and_select_structures,
)
from .walkforward import (
    compute_live_signals,
    newey_west_tstat,
    run_purged_walkforward,
    select_purged_training_rows,
)

__version__ = "1.0.0"

__all__ = [
    "ACTIVE_SLEEVE_MARKETS",
    "CASH_RESERVE_SHARE_PCT",
    "DAY1_DEPLOYED_SHARE_PCT",
    "DataPaths",
    "IV_RANK_SHORTLIST_MAX_PCT",
    "MANDATE_SLEEVE_SHARES_PCT",
    "MANDATE_TARGET_VOL_PCT",
    "PHYSICAL_SD_OVERLAYS",
    "UNIVERSE_META",
    "WALKFORWARD_UNIVERSE",
    "add_cross_commodity_and_macro_features",
    "black76_option",
    "build_mandate_portfolio_and_scenarios",
    "build_weekly_market_panel",
    "compute_contract_matched_returns",
    "compute_cot_tradeable_availability",
    "compute_cross_market_correlation",
    "compute_live_signals",
    "compute_monthly_sst_availability",
    "evaluate_two_step_option_filter",
    "forecast_volatility_and_select_structures",
    "get_physical_sd_overlay",
    "load_macro_fx_enso_cot",
    "newey_west_tstat",
    "run_purged_walkforward",
    "select_purged_training_rows",
]
