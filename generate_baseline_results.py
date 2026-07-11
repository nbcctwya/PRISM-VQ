#!/usr/bin/env python
"""Generate Baseline Results Protocol v1.0 artifacts from existing predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_results.config import BASELINE_ID, MODEL_ID, RUNS, SEEDS, SPLITS, TEST, prediction_path
from baseline_results.metrics import PORTFOLIO_METRICS, RANKING_METRICS, curve_from_report, portfolio_metrics, ranking_metrics

SEED_COLUMNS = ["market", "model", "seed", *RANKING_METRICS, *PORTFOLIO_METRICS,
                "num_test_days", "pred_path_or_ckpt_path"]
ENSEMBLE_COLUMNS = ["market", "model", "ensemble_method", *RANKING_METRICS, *PORTFOLIO_METRICS,
                    "num_test_days", "seeds", "pred_paths"]
AGG_COLUMNS = ["market", "model", *[f"{metric}_{stat}" for metric in (*RANKING_METRICS, *PORTFOLIO_METRICS)
                                         for stat in ("mean", "std")]]
STRATEGY = {"class": "TopkDropoutStrategy", "topk": 30, "n_drop": 5, "method_sell": "bottom",
            "method_buy": "top", "hold_thresh": 1, "only_tradable": False,
            "forbid_all_trade_at_limit": True, "risk_degree": 0.95}
BACKTEST = {"freq": "day", "account": 100000000, "open_cost": 0.0005,
            "close_cost": 0.0015, "min_cost": 0, "deal_price": "close"}


def load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_pickle(path).copy()
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.names != ["datetime", "instrument"]:
        raise ValueError(f"{path}: expected (datetime, instrument) MultiIndex")
    frame.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(frame.index.get_level_values("datetime")), frame.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"])
    if frame.index.duplicated().any() or not {"score", "label"}.issubset(frame.columns):
        raise ValueError(f"{path}: duplicate index or missing score/label")
    return frame[["score", "label"]].sort_index()


def make_ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    scores = pd.concat([f["score"].rename(f"score_{i}") for i, f in enumerate(frames)], axis=1, join="inner")
    labels = pd.concat([f["label"].rename(f"label_{i}") for i, f in enumerate(frames)], axis=1, join="inner")
    comparable = labels.dropna(how="all")
    if not comparable.apply(lambda row: row.dropna().nunique() <= 1, axis=1).all():
        raise ValueError("seed labels differ after alignment")
    return pd.DataFrame({"score": scores.mean(axis=1), "label": labels.bfill(axis=1).iloc[:, 0]})


def check_calendar_coverage(market: str, signal: pd.Series) -> None:
    from qlib.data import D
    calendar = pd.DatetimeIndex(D.calendar(start_time=TEST[0], end_time=TEST[1], freq="day"))
    dates = pd.DatetimeIndex(signal.index.get_level_values("datetime").unique())
    missing = calendar.difference(dates)
    if len(missing):
        raise ValueError(f"{market}: predictions miss {len(missing)} test trading days: {missing[:5].tolist()}")


def run_backtest(market: str, signal: pd.Series) -> pd.DataFrame:
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy import TopkDropoutStrategy

    cfg = RUNS[market]
    strategy = TopkDropoutStrategy(
        signal=signal, topk=30, n_drop=5, method_sell="bottom", method_buy="top", hold_thresh=1,
        only_tradable=False, forbid_all_trade_at_limit=True, risk_degree=0.95)
    report, _ = backtest_daily(
        start_time=TEST[0], end_time=TEST[1], strategy=strategy,
        executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}},
        account=100000000, benchmark=cfg["benchmark"],
        exchange_kwargs={"freq": "day", "deal_price": "close", "open_cost": 0.0005,
                         "close_cost": 0.0015, "min_cost": 0})
    return report


def init_qlib(market: str) -> None:
    import qlib
    from qlib.constant import REG_CN, REG_US
    cfg = RUNS[market]
    qlib.init(provider_uri=str(Path(cfg["provider_uri"]).expanduser()), region=REG_CN if cfg["region"] == "cn" else REG_US)


def metric_row(market: str, frame: pd.DataFrame, report: pd.DataFrame) -> dict:
    curve = curve_from_report(report)
    return {"market": market, "model": MODEL_ID, **ranking_metrics(frame),
            **portfolio_metrics(curve["daily_ret_net"])}


def aggregate(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (market, model), group in seed_df.groupby(["market", "model"], sort=True):
        row = {"market": market, "model": model}
        for metric in (*RANKING_METRICS, *PORTFOLIO_METRICS):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows, columns=AGG_COLUMNS)


def display_table(numeric: pd.DataFrame) -> pd.DataFrame:
    out = numeric[["market", "model"]].copy()
    for metric in (*RANKING_METRICS, *PORTFOLIO_METRICS):
        out[metric] = numeric.apply(lambda r: f"{r[f'{metric}_mean']:.4f} ± {r[f'{metric}_std']:.4f}", axis=1)
    return out


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return None


def write_metadata(root: Path, out: Path, qlib_version: str) -> None:
    markets = {}
    for market, cfg in RUNS.items():
        markets[market] = {k: v for k, v in cfg.items() if k != "run_dir"} | {
            "provider_uri": cfg["provider_uri"], "deal_price": "close", "trade_unit": "Qlib region default",
            "executor": "SimulatorExecutor(time_per_step=day)", "long_only": True, "leverage": False,
            "test_start": TEST[0], "test_end": TEST[1]}
    eval_cfg = {
        "schema_version": "1.0", "baseline": BASELINE_ID, "models": [MODEL_ID], "seeds": SEEDS,
        "markets_or_datasets": list(RUNS), "splits": SPLITS, "label_horizon": "5-day forward return: close(t+5)/close(t+1)-1",
        "strategy": STRATEGY | {"freq": "day"}, "backtest": BACKTEST | {"start_time": TEST[0], "end_time": TEST[1]},
        "markets": markets, "qlib_version": qlib_version,
        "signal_alignment": {"signal_date": "t-1", "trade_date": "t", "qlib_internal_shift": 1,
                             "adapter_manual_shift": False},
        "return_semantics": {"qlib_report.return": "gross_before_cost", "net_formula": "report.return - report.cost",
                             "cost_deducted_exactly_once": True,
                             "basis": "Qlib portfolio report exposes return and cost separately; native excess_return_without_cost subtracts bench from return"},
        "metric_convention": {"annualization": 252, "returns": "log1p(daily net simple return)", "std_ddof": 1,
                              "risk_free_rate": 0, "MAR_daily": 0,
                              "AR": "exp(mean(g)*252)-1", "STD": "std(g,ddof=1)*sqrt(252)",
                              "MDD": "min([1,exp(cumsum(g))]/cummax-1)",
                              "Sharpe": "sqrt(252)*mean(g)/std(g,ddof=1)",
                              "Sortino": "sqrt(252)*mean(g)/sqrt(mean(min(g,0)^2))", "Calmar": "AR/abs(MDD)"},
        "ensemble": {"enabled": True, "methods": ["avg_none"], "join": "inner", "normalize": "none",
                     "score_formula": "arithmetic mean of aligned raw seed scores",
                     "ranking_metrics_source": "ensemble score joined to aligned test label and recomputed"},
        "data_version_or_cutoff": "Qlib local data covering through 2025-12-31; repository does not expose a dataset revision ID",
        "git_commit": git_commit(root),
    }
    (out / "metadata" / "eval_config.json").write_text(json.dumps(eval_cfg, indent=2, ensure_ascii=False) + "\n")
    files = {"seed_metrics": "metrics/seed_metrics.csv", "aggregate_metrics": "metrics/aggregate_metrics.csv",
             "seed_table": "tables/seed_mean_std.csv", "eval_config": "metadata/eval_config.json",
             "validation": "diagnostics/validation.json", "ensemble_metrics": "metrics/ensemble_metrics.csv",
             "ensemble_table": "tables/ensemble.csv", "ensemble_curves": "curves/ensemble/*.csv",
             "ensemble_scores": "artifacts/ensemble/*.pkl"}
    manifest = {"schema_version": "1.0", "baseline": BASELINE_ID,
                "description": "PRISM-VQ predictions evaluated with Baseline Results Protocol v1.0",
                "primary_keys": {"seed_metrics": ["market", "model", "seed"],
                                 "aggregate_metrics": ["market", "model"],
                                 "ensemble_metrics": ["market", "model", "ensemble_method"]}, "files": files}
    (out / "metadata" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.out if args.out.is_absolute() else root / args.out
    for sub in ("metrics", "tables", "curves/ensemble", "artifacts/ensemble", "metadata", "diagnostics"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    import qlib
    seed_rows, ensemble_rows = [], []
    for market in RUNS:
        init_qlib(market)
        frames = []
        rel_paths = []
        for seed in SEEDS:
            path = prediction_path(root, market, seed)
            if not path.exists():
                raise FileNotFoundError(f"missing configured experiment: {path}")
            frame = load_prediction(path)
            check_calendar_coverage(market, frame["score"])
            report = run_backtest(market, frame["score"])
            row = metric_row(market, frame, report) | {"seed": seed, "pred_path_or_ckpt_path": path.relative_to(root).as_posix()}
            seed_rows.append(row)
            frames.append(frame)
            rel_paths.append(path.relative_to(root).as_posix())

        ensemble = make_ensemble(frames)
        check_calendar_coverage(market, ensemble["score"])
        score_path = out / "artifacts" / "ensemble" / f"{market}_{MODEL_ID}_avg_none.pkl"
        ensemble.to_pickle(score_path)
        report = run_backtest(market, ensemble["score"])
        curve = curve_from_report(report)
        curve.to_csv(out / "curves" / "ensemble" / f"{market}_{MODEL_ID}.csv", index=False)
        row = metric_row(market, ensemble, report) | {"ensemble_method": "avg_none",
              "seeds": ",".join(map(str, SEEDS)), "pred_paths": ",".join(rel_paths)}
        ensemble_rows.append(row)

    seed_df = pd.DataFrame(seed_rows, columns=SEED_COLUMNS).sort_values(["market", "model", "seed"])
    agg_df = aggregate(seed_df)
    ens_df = pd.DataFrame(ensemble_rows, columns=ENSEMBLE_COLUMNS).sort_values(["market", "model", "ensemble_method"])
    seed_df.to_csv(out / "metrics" / "seed_metrics.csv", index=False)
    agg_df.to_csv(out / "metrics" / "aggregate_metrics.csv", index=False)
    ens_df.to_csv(out / "metrics" / "ensemble_metrics.csv", index=False)
    display_table(agg_df).to_csv(out / "tables" / "seed_mean_std.csv", index=False)
    ens_display = ens_df.copy()
    for metric in (*RANKING_METRICS, *PORTFOLIO_METRICS):
        ens_display[metric] = ens_display[metric].map(lambda x: f"{x:.4f}")
    ens_display.to_csv(out / "tables" / "ensemble.csv", index=False)
    write_metadata(root, out, qlib.__version__)
    print(f"Generated protocol artifacts in {out}")


if __name__ == "__main__":
    main()
