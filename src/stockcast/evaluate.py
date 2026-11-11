"""Walk-forward evaluation of two framings of the same problem.

Experiment 1 predicts tomorrow's closing *price*. It produces an R-squared
near 1.0, and so does the rule "tomorrow equals today".

Experiment 2 predicts the next week's *return*, which is what a forecast
would actually have to get right to be useful, and compares against honest
baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .features import HORIZON, feature_columns


def models(k_best: int = 12, seed: int = 0) -> dict:
    """Each model sits behind the same SelectKBest step, fitted inside every fold."""
    def select():
        return SelectKBest(f_regression, k=k_best)

    return {
        "Ridge": make_pipeline(StandardScaler(), select(), Ridge(alpha=10.0)),
        "Random Forest": make_pipeline(select(), RandomForestRegressor(
            n_estimators=200, min_samples_leaf=50, max_features=0.5, n_jobs=-1, random_state=seed)),
        "Bagging": make_pipeline(select(), BaggingRegressor(
            DecisionTreeRegressor(min_samples_leaf=50), n_estimators=100, n_jobs=-1,
            random_state=seed)),
        "Gradient Boosting": make_pipeline(select(), GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.03, max_depth=3, subsample=0.7,
            min_samples_leaf=50, random_state=seed)),
    }


def walk_forward_splits(dates: pd.Series, n_folds: int = 6, gap: int = HORIZON):
    """Expanding-window folds split by calendar date.

    A gap of ``HORIZON`` trading days separates train from test so that no
    training target overlaps the test period.
    """
    days = np.sort(dates.unique())
    first_test = len(days) // 3  # always train on at least a third of history
    edges = np.linspace(first_test, len(days), n_folds + 1).astype(int)
    for start, stop in zip(edges[:-1], edges[1:]):
        train = dates < days[start - gap]
        test = (dates >= days[start]) & (dates <= days[stop - 1])
        yield train.to_numpy(), test.to_numpy()


def price_level_experiment(data: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Predict tomorrow's close, per ticker, and compare with "no change"."""
    cols = feature_columns(data)
    rows = []
    for ticker, df in data.groupby("ticker"):
        df = df.reset_index(drop=True)
        split = int(len(df) * 0.8)
        train, test = df.iloc[:split - 1], df.iloc[split:]
        # Predict the day-ahead change and add it to today's close: tree models
        # cannot extrapolate to price levels they have never seen.
        model = make_pipeline(SelectKBest(f_regression, k=12), GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.03, max_depth=3, subsample=0.7, random_state=seed))
        model.fit(train[cols], train["target_next_close"] / train["close"] - 1)
        pred = test["close"] * (1 + model.predict(test[cols]))
        actual = test["target_next_close"]
        rows.append({
            "ticker": ticker,
            "r2_model": r2_score(actual, pred),
            "r2_naive": r2_score(actual, test["close"]),
            "rmse_model": float(np.sqrt(mean_squared_error(actual, pred))),
            "rmse_naive": float(np.sqrt(mean_squared_error(actual, test["close"]))),
        })
    return pd.DataFrame(rows)


def return_experiment(data: pd.DataFrame, n_folds: int = 6, seed: int = 0):
    """Walk-forward prediction of the next-week return, pooled across tickers."""
    cols = feature_columns(data)
    x, y = data[cols], data["target_return"]
    preds = data[["date", "ticker", "target_return"]].copy()
    fold_rows, selected = [], {}
    for fold, (train, test) in enumerate(walk_forward_splits(data["date"], n_folds), start=1):
        preds.loc[test, "fold"] = fold
        preds.loc[test, "Historical mean"] = y[train].mean()
        for name, model in models(seed=seed).items():
            model.fit(x[train], y[train])
            preds.loc[test, name] = model.predict(x[test])
            if name == "Gradient Boosting":
                mask = model.named_steps["selectkbest"].get_support()
                for c in np.array(cols)[mask]:
                    selected[c] = selected.get(c, 0) + 1
        fold_rows.append({"fold": fold, "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                          "test_start": data.loc[test, "date"].min().date(),
                          "test_end": data.loc[test, "date"].max().date()})
    preds = preds.dropna(subset=["fold"])
    return preds, pd.DataFrame(fold_rows), selected


def _rank_correlation(preds: pd.DataFrame, name: str) -> float:
    """Mean daily Spearman correlation between predicted and realised returns across tickers."""
    if preds.groupby("date")[name].nunique().max() <= 1:
        return float("nan")  # a constant forecast cannot rank stocks
    ranks = preds.groupby("date")[[name, "target_return"]].rank()
    daily = ranks.groupby(preds["date"]).corr().xs(name, level=1)["target_return"]
    return float(daily.mean())


def score_returns(preds: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample metrics for each model and baseline."""
    actual = preds["target_return"]
    up = actual > 0
    names = [c for c in preds.columns if c not in ("date", "ticker", "target_return", "fold")]
    rows = []
    for name in names:
        p = preds[name]
        # R2 against the historical-mean forecast: positive means real skill.
        r2_oos = 1 - ((actual - p) ** 2).sum() / ((actual - preds["Historical mean"]) ** 2).sum()
        rows.append({
            "model": name,
            "rmse": float(np.sqrt(mean_squared_error(actual, p))),
            "r2_vs_mean": float(r2_oos),
            "direction_accuracy": float(((p > 0) == up).mean()),
            "rank_correlation": _rank_correlation(preds, name),
        })
    rows.append({"model": "Always predict up", "rmse": np.nan, "r2_vs_mean": np.nan,
                 "direction_accuracy": float(up.mean()), "rank_correlation": np.nan})
    return pd.DataFrame(rows)


def backtest(preds: pd.DataFrame, model: str, top_n: int = 3, cost_bps: float = 5.0) -> pd.DataFrame:
    """Weekly long-only strategy: hold the ``top_n`` stocks the model ranks highest.

    Rebalances every ``HORIZON`` days so holding periods do not overlap, and
    charges ``cost_bps`` per unit of turnover. Compared with holding all ten
    stocks equally.
    """
    days = np.sort(preds["date"].unique())[::HORIZON]
    weekly = preds[preds["date"].isin(days)]
    rows, previous = [], set()
    for day, g in weekly.groupby("date"):
        picks = set(g.nlargest(top_n, model)["ticker"])
        turnover = len(picks - previous) / top_n
        gross = g.loc[g["ticker"].isin(picks), "target_return"].mean()
        rows.append({"date": day, "strategy": gross - turnover * 2 * cost_bps / 1e4,
                     "equal_weight": g["target_return"].mean()})
        previous = picks
    out = pd.DataFrame(rows).set_index("date")
    return out


def performance(returns: pd.Series) -> dict:
    periods = 252 / HORIZON
    growth = (1 + returns).cumprod()
    return {
        "annual_return": float(growth.iloc[-1] ** (periods / len(returns)) - 1),
        "annual_volatility": float(returns.std() * np.sqrt(periods)),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(periods)),
        "max_drawdown": float((growth / growth.cummax() - 1).min()),
    }
