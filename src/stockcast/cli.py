"""Run both experiments and write results and figures.

    python -m stockcast             # uses cached prices if present
    python -m stockcast --refresh   # download prices again
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import load_prices  # noqa: E402
from .evaluate import (  # noqa: E402
    backtest,
    performance,
    price_level_experiment,
    return_experiment,
    score_returns,
)
from .features import build_features  # noqa: E402

INK, ACCENT, MUTED, WARN = "#14213d", "#0f766e", "#b6bcc6", "#b45309"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")
    ax.tick_params(colors="#4b5563", labelsize=9)
    ax.grid(axis="y", color="#eef0f3", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_illusion(levels, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=160)
    x = range(len(levels))
    ax.bar([i - 0.2 for i in x], levels["r2_naive"], width=0.38, color=MUTED,
           label='Naive: "tomorrow = today"')
    ax.bar([i + 0.2 for i in x], levels["r2_model"], width=0.38, color=ACCENT,
           label="Gradient boosting")
    ax.set_xticks(list(x), levels["ticker"])
    ax.set_ylim(min(0.9, levels[["r2_naive", "r2_model"]].min().min() - 0.01), 1.0)
    ax.set_ylabel("R-squared on next-day price", color="#4b5563")
    ax.set_title("Predicting price level: a model is no better than doing nothing", loc="left",
                 color=INK, fontweight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_returns(scores, path: Path) -> None:
    s = scores.dropna(subset=["r2_vs_mean"]).query("model != 'Historical mean'")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3.4), dpi=160)
    bars = a.barh(s["model"], s["r2_vs_mean"] * 100, color=[ACCENT if v > 0 else WARN
                                                             for v in s["r2_vs_mean"]], height=0.55)
    a.bar_label(bars, fmt="%+.2f%%", fontsize=8, padding=3, color=INK)
    a.axvline(0, color=INK, lw=1)
    a.set_xlabel("Out-of-sample R-squared vs historical mean", color="#4b5563", fontsize=8)
    a.set_title("Skill at predicting weekly returns", loc="left", color=INK, fontweight="bold",
                fontsize=10)
    a.margins(x=0.3)
    d = scores[scores["model"] != "Historical mean"]
    bars = b.barh(d["model"], d["direction_accuracy"] * 100,
                  color=[MUTED if m == "Always predict up" else ACCENT for m in d["model"]], height=0.55)
    b.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=3, color=INK)
    b.set_xlim(45, 62)
    b.set_xlabel("Direction called correctly", color="#4b5563", fontsize=8)
    b.set_title("Direction accuracy", loc="left", color=INK, fontweight="bold", fontsize=10)
    for ax in (a, b):
        _style(ax)
        ax.grid(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_backtest(bt, model: str, path: Path) -> None:
    growth = (1 + bt).cumprod()
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=160)
    ax.plot(growth.index, growth["equal_weight"], color=MUTED, lw=2, label="Hold all 10 equally")
    ax.plot(growth.index, growth["strategy"], color=ACCENT, lw=2,
            label=f"Top 3 by {model}, after costs")
    ax.set_ylabel("Growth of $1", color="#4b5563")
    ax.set_title("Out-of-sample backtest", loc="left", color=INK, fontweight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock trend forecasting experiments.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    args.docs.mkdir(exist_ok=True)

    prices = load_prices(refresh=args.refresh)
    data = build_features(prices)
    print(f"{len(data):,} rows, {data['ticker'].nunique()} tickers, "
          f"{data['date'].min().date()} to {data['date'].max().date()}")

    levels = price_level_experiment(data)
    print("\nExperiment 1: next-day price level\n", levels.round(4).to_string(index=False))

    preds, folds, selected = return_experiment(data)
    scores = score_returns(preds)
    print("\nExperiment 2: next-week return (walk-forward)\n", scores.round(4).to_string(index=False))

    best = scores.dropna(subset=["rank_correlation"]).sort_values("rank_correlation").iloc[-1]["model"]
    bt = backtest(preds, best)
    perf = {k: performance(bt[k]) for k in bt.columns}
    print(f"\nBacktest using {best}:", json.dumps(perf, indent=2))

    plot_illusion(levels, args.docs / "price_level_illusion.png")
    plot_returns(scores, args.docs / "return_skill.png")
    plot_backtest(bt, best, args.docs / "backtest.png")
    levels.to_csv(args.out / "price_level.csv", index=False)
    scores.to_csv(args.out / "return_scores.csv", index=False)
    folds.to_csv(args.out / "folds.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps({
        "rows": len(data), "test_rows": len(preds), "folds": len(folds),
        "price_level_mean_r2": levels[["r2_model", "r2_naive"]].mean().to_dict(),
        "backtest_model": best, "backtest": perf,
        "features_selected_in_folds": dict(sorted(selected.items(), key=lambda kv: -kv[1])),
    }, indent=2))


if __name__ == "__main__":
    main()
