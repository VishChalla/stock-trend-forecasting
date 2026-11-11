"""Download and cache daily prices from Yahoo Finance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TICKERS = ("AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "XOM", "JNJ", "WMT", "PG")
MARKET = "SPY"
START, END = "2016-01-01", "2026-10-31"  # fixed window so results are reproducible


def load_prices(cache: Path = Path("data/prices.csv"), refresh: bool = False) -> pd.DataFrame:
    """Long table of adjusted daily bars: date, ticker, open, high, low, close, volume."""
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["date"])

    import yfinance as yf

    raw = yf.download([*TICKERS, MARKET], start=START, end=END, auto_adjust=True,
                      progress=False, group_by="ticker")
    frames = []
    for ticker in (*TICKERS, MARKET):
        bars = raw[ticker].dropna().reset_index()
        bars.columns = [c.lower() for c in bars.columns]
        bars.insert(1, "ticker", ticker)
        frames.append(bars)
    prices = pd.concat(frames, ignore_index=True).round(4)
    cache.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache, index=False)
    return prices
