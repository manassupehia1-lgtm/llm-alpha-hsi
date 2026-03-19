"""
Add backtrader backtesting engine with LLM alpha and momentum baseline strategies-----------
backtrader-based backtesting engine for the LLM-driven long/short alpha strategy.

Features:
- Equal-weighted long/short portfolio, monthly rebalancing
- 15 bps transaction cost per leg (realistic for HK equities)
- Benchmark comparison: HSI total return index, naive 6M momentum baseline
- Sharpe ratio, Calmar ratio, max drawdown analytics

Usage:
    python src/backtest.py --signals data/processed/signals.csv
"""

import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.analyzers as btanalyzers

from utils import setup_logging, load_config

logger = logging.getLogger(__name__)


class PandasSignalFeed(btfeeds.PandasData):
    """PandasData feed extended with a 'signal' line for the strategy."""
    lines = ("signal",)
    params = (("signal", -1),)


class LLMAlphaStrategy(bt.Strategy):
    """
    Long/short strategy driven by XGBoost + GPT-4 sentiment signals.
    Rebalances on the first trading day of each month.
    Goes long top-signal stocks, short bottom-signal stocks, equal weight.
    """
    params = (
        ("max_long", 10),
        ("max_short", 10),
        ("verbose", False),
    )

    def __init__(self):
        self.month_counter = -1

    def _should_rebalance(self):
        current_month = self.datas[0].datetime.date(0).month
        if current_month != self.month_counter:
            self.month_counter = current_month
            return True
        return False

    def next(self):
        if not self._should_rebalance():
            return

        longs = [d for d in self.datas if d.signal[0] == 1][:self.params.max_long]
        shorts = [d for d in self.datas if d.signal[0] == -1][:self.params.max_short]

        # Close all existing positions first
        for d in self.datas:
            if self.getposition(d).size != 0:
                self.close(data=d)

        pv = self.broker.getvalue()
        if longs:
            w = 0.5 / len(longs)
            for d in longs:
                self.buy(data=d, size=(pv * w) / d.close[0])
        if shorts:
            w = 0.5 / len(shorts)
            for d in shorts:
                self.sell(data=d, size=(pv * w) / d.close[0])


class MomentumBaseline(bt.Strategy):
    """Naive 6-month price momentum long/short baseline."""
    params = (("lookback", 126), ("n_long", 10), ("n_short", 10))

    def __init__(self):
        self.month_counter = -1
        self.mom = {d: bt.ind.ROC(d.close, period=self.params.lookback) for d in self.datas}

    def _should_rebalance(self):
        current_month = self.datas[0].datetime.date(0).month
        if current_month != self.month_counter:
            self.month_counter = current_month
            return True
        return False

    def next(self):
        if not self._should_rebalance():
            return

        vals = {d: self.mom[d][0] for d in self.datas if self.mom[d][0] == self.mom[d][0]}
        if not vals:
            return

        ranked = sorted(vals, key=vals.get, reverse=True)
        longs, shorts = ranked[:self.params.n_long], ranked[-self.params.n_short:]

        for d in self.datas:
            if self.getposition(d).size != 0:
                self.close(data=d)

        pv = self.broker.getvalue()
        for d in longs:
            self.buy(data=d, size=(pv * 0.05) / d.close[0])
        for d in shorts:
            self.sell(data=d, size=(pv * 0.05) / d.close[0])


def compute_performance_metrics(returns: pd.Series, benchmark: pd.Series) -> dict:
    """Compute Sharpe, Calmar, max drawdown, hit rate, and information ratio."""
    ann = 252
    ann_ret = returns.mean() * ann
    ann_vol = returns.std() * np.sqrt(ann)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + returns).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    hit_rate = (returns > 0).mean()

    excess = returns - benchmark.reindex(returns.index).fillna(0)
    excess_ann_vol = excess.std() * np.sqrt(ann)
    ir = (excess.mean() * ann) / excess_ann_vol if excess_ann_vol > 0 else 0.0

    return {
        "ann_return": round(ann_ret, 4),
        "ann_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(calmar, 4),
        "hit_rate": round(hit_rate, 4),
        "information_ratio": round(ir, 4),
        "total_return": round(cum.iloc[-1] - 1, 4),
    }


def run_strategy(StrategyClass, strategy_kwargs, data_feeds, initial_cash, commission_bps):
    """Set up and run a single backtrader cerebro instance."""
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission_bps / 10000)
    for data in data_feeds:
        cerebro.adddata(data)
    cerebro.addstrategy(StrategyClass, **strategy_kwargs)
    cerebro.addanalyzer(btanalyzers.TimeReturn, _name="time_return", timeframe=bt.TimeFrame.Days)
    result = cerebro.run()
    daily_rets = pd.Series(result[0].analyzers.time_return.get_analysis())
    daily_rets.index = pd.to_datetime(daily_rets.index)
    return daily_rets


def run_backtest(signals_path, prices_path, hsi_returns_path, output_dir,
                 initial_cash=10_000_000, commission_bps=15):
    """Run full backtest for strategy and momentum baseline; print summary."""
    logger.info("Loading data...")
    signals_df = pd.read_csv(signals_path, parse_dates=["date"])
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    hsi_returns = pd.read_csv(hsi_returns_path, index_col=0, parse_dates=True).squeeze()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tickers = signals_df["ticker"].unique()
    results = {}

    for strat_name, StratClass, skwargs in [
        ("llm_alpha", LLMAlphaStrategy, {}),
        ("momentum_baseline", MomentumBaseline, {}),
    ]:
        logger.info(f"Running: {strat_name}")
        feeds = []
        for ticker in tickers:
            tp = prices[[ticker]].dropna()
            ts = signals_df[signals_df["ticker"] == ticker][["date","predicted_signal"]].set_index("date")
            merged = tp.join(ts, how="left")
            merged.columns = ["close", "signal"]
            merged[["open","high","low"]] = merged[["close","close","close"]]
            merged["volume"] = 1_000_000
            merged["signal"] = merged["signal"].fillna(0)
            feeds.append(PandasSignalFeed(dataname=merged, name=ticker))

        daily_rets = run_strategy(StratClass, skwargs, feeds, initial_cash, commission_bps)
        perf = compute_performance_metrics(daily_rets, hsi_returns)
        results[strat_name] = perf
        daily_rets.to_csv(out / f"returns_{strat_name}.csv")
        logger.info(f"  {strat_name}: Sharpe={perf['sharpe_ratio']:.3f} MaxDD={perf['max_drawdown']:.2%}")

    summary = pd.DataFrame(results).T
    logger.info("\n" + "="*60 + "\nBACKTEST RESULTS\n" + "="*60)
    logger.info("\n" + summary.to_string())
    summary.to_csv(out / "performance_summary.csv")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--signals", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    run_backtest(
        signals_path=args.signals,
        prices_path=cfg["data"]["prices_path"],
        hsi_returns_path=cfg["data"]["hsi_returns_path"],
        output_dir=cfg.get("output_dir", "results/"),
        initial_cash=cfg["backtest"].get("initial_cash", 10_000_000),
        commission_bps=cfg["backtest"].get("commission_bps", 15),
    )


if __name__ == "__main__":
    main()
