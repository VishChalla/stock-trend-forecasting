"""Tests run on generated random-walk prices, so they need no network."""

import numpy as np
import pandas as pd
import pytest

from stockcast.evaluate import (
    backtest,
    performance,
    price_level_experiment,
    return_experiment,
    score_returns,
    walk_forward_splits,
)
from stockcast.features import HORIZON, build_features, feature_columns


def random_walk_prices(n_days=900, tickers=("AAA", "BBB", "CCC", "SPY"), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    frames = []
    for t in tickers:
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days)))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": t, "open": close * 0.999, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": rng.integers(1e6, 5e6, n_days)}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def data():
    return build_features(random_walk_prices())


def test_features_have_no_missing_values_and_exclude_the_market(data):
    assert not data.isna().any().any()
    assert set(data["ticker"]) == {"AAA", "BBB", "CCC"}
    assert "target_return" not in feature_columns(data)


def test_features_do_not_look_ahead():
    """Changing future prices must not change today's features."""
    prices = random_walk_prices()
    cutoff = prices["date"].sort_values().unique()[600]
    changed = prices.copy()
    changed.loc[changed["date"] > cutoff, ["open", "high", "low", "close"]] *= 3
    a, b = build_features(prices), build_features(changed)
    cols = feature_columns(a)
    a, b = a[a["date"] <= cutoff], b[b["date"] <= cutoff]
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


def test_target_is_the_forward_return(data):
    one = data[data["ticker"] == "AAA"].reset_index(drop=True)
    expected = one["close"].iloc[HORIZON] / one["close"].iloc[0] - 1
    assert one["target_return"].iloc[0] == pytest.approx(expected)


def test_walk_forward_never_trains_on_the_future(data):
    for train, test in walk_forward_splits(data["date"], n_folds=4):
        train_days = np.sort(data.loc[train, "date"].unique())
        test_days = np.sort(data.loc[test, "date"].unique())
        all_days = np.sort(data["date"].unique())
        gap = np.searchsorted(all_days, test_days[0]) - np.searchsorted(all_days, train_days[-1])
        assert gap > HORIZON  # no training target overlaps the test window


def test_models_find_no_skill_in_a_random_walk(data):
    """On pure noise an honest evaluation must report roughly zero skill."""
    preds, _, _ = return_experiment(data, n_folds=3)
    scores = score_returns(preds).set_index("model")
    assert scores.loc["Gradient Boosting", "r2_vs_mean"] < 0.02
    assert abs(scores.loc["Gradient Boosting", "direction_accuracy"] - 0.5) < 0.08


def test_naive_forecast_matches_the_model_on_price_level(data):
    """On a random walk, "tomorrow = today" is the best possible price forecast."""
    levels = price_level_experiment(data)
    assert (levels["rmse_naive"] <= levels["rmse_model"] * 1.02).all()


def test_backtest_and_performance(data):
    preds, _, _ = return_experiment(data, n_folds=3)
    bt = backtest(preds, "Ridge", top_n=1)
    assert set(bt.columns) == {"strategy", "equal_weight"}
    assert performance(pd.Series([0.01, 0.01, 0.01, 0.0]))["max_drawdown"] == 0
