"""
feature_engineering.py
-----------------------
Constructs the full feature matrix combining GPT-4 sentiment signals
with price/volume momentum indicators for HSI constituent stocks.

Outputs a panel dataset ready for XGBoost training.
"""

import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import setup_logging, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price / Volume Signal Construction
# ---------------------------------------------------------------------------

def compute_price_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multi-horizon price momentum signals.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format adjusted close prices (index=date, columns=ticker)
    
    Returns
    -------
    pd.DataFrame in long format with columns: ticker, date, mom_1m, mom_3m, mom_6m
    """
    records = []
    for horizon, periods in [("mom_1m", 21), ("mom_3m", 63), ("mom_6m", 126)]:
        ret = prices.pct_change(periods).shift(1)  # Shift 1 to avoid look-ahead
        melted = ret.reset_index().melt(id_vars="date", var_name="ticker", value_name=horizon)
        records.append(melted.set_index(["date", "ticker"]))
    
    return pd.concat(records, axis=1).reset_index()


def compute_volume_weighted_momentum(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Volume-weighted price momentum: sum(return_t * volume_t) over 1-month window.
    Higher volume on up-days boosts signal.
    """
    daily_ret = prices.pct_change()
    vwm = (daily_ret * volumes).rolling(21).sum() / volumes.rolling(21).sum()
    vwm = vwm.shift(1)
    return vwm.reset_index().melt(id_vars="date", var_name="ticker", value_name="vw_momentum")


def compute_volatility_adjusted_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """Sharpe-style momentum: 3M return / 3M rolling volatility."""
    ret_3m = prices.pct_change(63).shift(1)
    vol_3m = prices.pct_change().rolling(63).std().shift(1) * np.sqrt(252)
    adj_mom = ret_3m / vol_3m.replace(0, np.nan)
    return adj_mom.reset_index().melt(id_vars="date", var_name="ticker", value_name="vol_adj_mom")


def compute_52w_high_proximity(prices: pd.DataFrame) -> pd.DataFrame:
    """Ratio of current price to trailing 252-day high (George & Hwang 2004 factor)."""
    high_52w = prices.rolling(252).max().shift(1)
    proximity = prices / high_52w
    return proximity.reset_index().melt(id_vars="date", var_name="ticker", value_name="proximity_52w")


# ---------------------------------------------------------------------------
# Sentiment Feature Construction
# ---------------------------------------------------------------------------

def compute_sentiment_features(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Derive higher-order sentiment features from raw GPT-4 scores.
    
    Features:
    - raw composite score
    - 1-period score momentum (delta)
    - score surprise vs. trailing 4-period mean
    - cross-sectional z-score within HSI universe
    """
    scores = scores.sort_values(["ticker", "date"]).copy()
    scores["date"] = pd.to_datetime(scores["date"])
    
    # 1-period score momentum
    scores["sentiment_delta"] = scores.groupby("ticker")["composite_score"].diff(1)
    
    # Score surprise: deviation from trailing 4-period rolling mean
    trailing_mean = scores.groupby("ticker")["composite_score"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).mean()
    )
    scores["sentiment_surprise"] = scores["composite_score"] - trailing_mean
    
    # Cross-sectional z-score (within each date)
    scores["sentiment_zscore"] = scores.groupby("date")["composite_score"].transform(
        lambda x: (x - x.mean()) / x.std().replace(0, np.nan)
    )
    
    return scores


# ---------------------------------------------------------------------------
# Target Construction
# ---------------------------------------------------------------------------

def compute_forward_returns(prices: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """
    Compute forward 1-month returns for each ticker.
    
    IMPORTANT: forward returns are shifted -horizon to avoid look-ahead.
    These are only used during training; never during live signal generation.
    """
    fwd_ret = prices.pct_change(horizon).shift(-horizon)
    return fwd_ret.reset_index().melt(id_vars="date", var_name="ticker", value_name="fwd_return_1m")


def compute_excess_returns(fwd_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional excess return (subtract universe mean)."""
    fwd_returns["excess_return"] = fwd_returns.groupby("date")["fwd_return_1m"].transform(
        lambda x: x - x.mean()
    )
    return fwd_returns


def build_quintile_signal(df: pd.DataFrame, return_col: str = "excess_return") -> pd.DataFrame:
    """
    Assign long/short labels based on quintile ranking within each date.
    - Top 2 quintiles (Q4, Q5) -> label = 1 (long)
    - Bottom 2 quintiles (Q1, Q2) -> label = -1 (short)
    - Middle quintile (Q3) -> label = NaN (excluded)
    """
    df["quintile"] = df.groupby("date")[return_col].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
    )
    
    conditions = [
        df["quintile"] >= 3,  # Top 2 quintiles
        df["quintile"] <= 1,  # Bottom 2 quintiles
    ]
    choices = [1, -1]
    df["signal"] = np.select(conditions, choices, default=np.nan)
    return df


# ---------------------------------------------------------------------------
# Master Feature Matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    prices_path: str,
    volumes_path: str,
    sentiment_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Build the full feature matrix, merge all signals, and save to CSV.
    """
    logger.info("Loading price and volume data...")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    volumes = pd.read_csv(volumes_path, index_col=0, parse_dates=True)
    
    logger.info("Computing price/volume signals...")
    mom = compute_price_momentum(prices)
    vwm = compute_volume_weighted_momentum(prices, volumes)
    va_mom = compute_volatility_adjusted_momentum(prices)
    prox = compute_52w_high_proximity(prices)
    fwd = compute_forward_returns(prices)
    fwd = compute_excess_returns(fwd)
    
    logger.info("Computing sentiment features...")
    scores = pd.read_csv(sentiment_path, parse_dates=["date"])
    scores = compute_sentiment_features(scores)
    
    logger.info("Merging feature matrix...")
    df = mom.copy()
    for other in [vwm, va_mom, prox, fwd]:
        df = df.merge(other, on=["date", "ticker"], how="inner")
    
    df = df.merge(
        scores[["ticker", "date", "composite_score", "sentiment_delta",
                "sentiment_surprise", "sentiment_zscore"]],
        on=["ticker", "date"],
        how="left",
    )
    
    # Interaction features
    df["sent_x_mom1m"] = df["composite_score"] * df["mom_1m"]
    df["sent_x_mom3m"] = df["composite_score"] * df["mom_3m"]
    df["sent_delta_x_mom1m"] = df["sentiment_delta"] * df["mom_1m"]
    
    # Build labels
    df = build_quintile_signal(df)
    
    # Drop rows with NaN labels or critical features
    df = df.dropna(subset=["signal", "composite_score", "mom_1m", "mom_3m"])
    
    logger.info(f"Feature matrix shape: {df.shape}")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved feature matrix to {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build feature matrix")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    setup_logging()
    cfg = load_config(args.config)
    
    build_feature_matrix(
        prices_path=cfg["data"]["prices_path"],
        volumes_path=cfg["data"]["volumes_path"],
        sentiment_path=cfg["data"]["sentiment_scores_path"],
        output_path=cfg["data"]["features_path"],
    )


if __name__ == "__main__":
    main()
