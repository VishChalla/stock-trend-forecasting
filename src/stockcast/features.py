"""Feature engineering. Every feature at day t uses data up to the close of day t only."""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 5  # trading days ahead for the return target


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _one_ticker(bars: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    df = bars.sort_values("date").reset_index(drop=True)
    close, ret = df["close"], df["close"].pct_change()
    out = df[["date", "ticker", "close"]].copy()

    for lag in (1, 2, 5, 10, 21, 63):
        out[f"ret_{lag}d"] = close.pct_change(lag)
    for window in (10, 50, 200):
        out[f"close_vs_ma{window}"] = close / close.rolling(window).mean() - 1
    out["ma50_vs_ma200"] = close.rolling(50).mean() / close.rolling(200).mean() - 1
    out["volatility_10d"] = ret.rolling(10).std()
    out["volatility_63d"] = ret.rolling(63).std()
    out["rsi_14"] = _rsi(close)
    out["range_pct"] = (df["high"] - df["low"]) / close
    out["volume_z"] = (df["volume"] - df["volume"].rolling(63).mean()) / df["volume"].rolling(63).std()
    out["dist_from_52w_high"] = close / close.rolling(252).max() - 1
    out["day_of_week"] = df["date"].dt.dayofweek

    out = out.merge(market, on="date", how="left")
    out["rel_strength_21d"] = out["ret_21d"] - out["mkt_ret_21d"]

    # Targets look forward; everything above looks back.
    out["target_next_close"] = close.shift(-1)
    out["target_return"] = close.shift(-HORIZON) / close - 1
    return out


def build_features(prices: pd.DataFrame, market_ticker: str = "SPY") -> pd.DataFrame:
    mkt = prices[prices["ticker"] == market_ticker].sort_values("date")
    mkt_ret = mkt["close"].pct_change()
    market = pd.DataFrame({
        "date": mkt["date"].to_numpy(),
        "mkt_ret_1d": mkt_ret.to_numpy(),
        "mkt_ret_5d": mkt["close"].pct_change(5).to_numpy(),
        "mkt_ret_21d": mkt["close"].pct_change(21).to_numpy(),
        "mkt_volatility_21d": mkt_ret.rolling(21).std().to_numpy(),
    })
    frames = [_one_ticker(bars, market)
              for ticker, bars in prices.groupby("ticker") if ticker != market_ticker]
    data = pd.concat(frames, ignore_index=True).dropna().sort_values(["date", "ticker"])
    return data.reset_index(drop=True)


def feature_columns(data: pd.DataFrame) -> list[str]:
    skip = {"date", "ticker", "close", "target_next_close", "target_return"}
    return [c for c in data.columns if c not in skip]
