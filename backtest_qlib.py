import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _load_calculate_table_metrics():
    metric_path = Path(__file__).resolve().parent / "utils" / "metric.py"
    spec = importlib.util.spec_from_file_location("prism_vq_metric", metric_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metric helper from {metric_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.calculate_table_metrics


def _fallback_calculate_table_metrics(series, period, name, target_return=0):
    if period is not None:
        if isinstance(period, int):
            series = series[series.index.year == int(period)].copy()
        elif isinstance(period, list):
            series = series.loc[period[0] : period[1]].copy()

    try:
        daily_log_returns = series["return"]
        cum_return = series["return"].cumsum()
    except Exception:
        daily_log_returns = series
        cum_return = series.cumsum()

    normal_cum_return = np.exp(cum_return)
    max_cumulative_returns = normal_cum_return.cummax()
    drawdown = (normal_cum_return - max_cumulative_returns) / (max_cumulative_returns + 1e-9)
    mdd = drawdown.min()

    annual_return = daily_log_returns.mean() * 252
    annual_std = daily_log_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_std

    downside_returns = daily_log_returns[daily_log_returns < target_return]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - target_return) / downside_std if downside_std != 0 else np.nan
    calmar_ratio = annual_return / abs(mdd) if mdd != 0 else np.nan

    turnover = series["turnover"].mean() if isinstance(series, pd.DataFrame) and "turnover" in series else np.nan

    result = {
        "Annualized Return": round(annual_return, 4),
        "Annual Std": round(annual_std, 4),
        "MDD": round(mdd, 4),
        "Sharpe Ratio": round(sharpe_ratio, 4),
        "Sortino Ratio": round(sortino_ratio, 4),
        "Calmar Ratio": round(calmar_ratio, 4),
        "Cumulative Returns": round(cum_return.iloc[-1], 4),
        "Turnover": round(turnover, 4) if pd.notna(turnover) else np.nan,
    }
    return pd.DataFrame.from_dict(result, orient="index", columns=[name])


_CALCULATE_TABLE_METRICS = None


def _get_calculate_table_metrics():
    global _CALCULATE_TABLE_METRICS
    if _CALCULATE_TABLE_METRICS is not None:
        return _CALCULATE_TABLE_METRICS
    try:
        _CALCULATE_TABLE_METRICS = _load_calculate_table_metrics()
    except Exception as exc:
        print(f"Warning: falling back to local calculate_table_metrics implementation: {exc}")
        _CALCULATE_TABLE_METRICS = _fallback_calculate_table_metrics
    return _CALCULATE_TABLE_METRICS


DATE_COLUMNS = ("datetime", "date", "time")
INSTRUMENT_COLUMNS = ("instrument", "stock_id", "symbol", "code")
SCORE_COLUMNS = ("score", "pred", "prediction", "pred_score", "Pred", "PRED")
LABEL_COLUMNS = ("label", "realized_return", "realized_ret", "return", "ret", "LABEL0")


UNIVERSE_SETTINGS = {
    "sp500": {
        "region": "US",
        "provider_uri": "~/.qlib/qlib_data/us_data",
        "benchmark": "^gspc",
    },
    "csi300": {
        "region": "CN",
        "provider_uri": "~/.qlib/qlib_data/cn_data",
        "benchmark": "SH000300",
    },
    "csi500": {
        "region": "CN",
        "provider_uri": "~/.qlib/qlib_data/cn_data",
        "benchmark": "SH000905",
    },
}


def _find_first(candidates, names) -> Optional[str]:
    name_set = set(names)
    lower_to_name = {str(name).lower(): name for name in names}
    for candidate in candidates:
        if candidate in name_set:
            return candidate
        lowered = candidate.lower()
        if lowered in lower_to_name:
            return lower_to_name[lowered]
    return None


def _normalize_prediction_frame(pred_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    pred = pd.read_pickle(pred_path)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame(name=pred.name or "score")
    if not isinstance(pred, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame or Series from {pred_path}, got {type(pred)!r}.")

    pred = pred.copy()

    if isinstance(pred.index, pd.MultiIndex):
        index_names = list(pred.index.names)
        date_level = _find_first(DATE_COLUMNS, index_names)
        instrument_level = _find_first(INSTRUMENT_COLUMNS, index_names)
        if date_level is None or instrument_level is None:
            if pred.index.nlevels == 2:
                date_level = index_names[0]
                instrument_level = index_names[1]
            else:
                raise ValueError(
                    "Prediction MultiIndex must include datetime/date and instrument levels. "
                    f"Found index names: {index_names}."
                )

        pred = pred.reset_index()
        pred = pred.rename(columns={date_level: "datetime", instrument_level: "instrument"})
    else:
        date_col = _find_first(DATE_COLUMNS, pred.columns)
        instrument_col = _find_first(INSTRUMENT_COLUMNS, pred.columns)
        if date_col is None or instrument_col is None:
            raise ValueError(
                "Prediction file must contain datetime/date and instrument columns, or a "
                "MultiIndex with those levels. "
                f"Found columns: {list(pred.columns)}; index names: {getattr(pred.index, 'names', None)}."
            )
        pred = pred.rename(columns={date_col: "datetime", instrument_col: "instrument"})

    score_col = _find_first(SCORE_COLUMNS, pred.columns)
    if score_col is None:
        raise ValueError(
            "Prediction file must contain a score column. Accepted names: "
            f"{list(SCORE_COLUMNS)}. Found columns: {list(pred.columns)}."
        )

    label_col = _find_first(LABEL_COLUMNS, pred.columns)
    if label_col is None:
        print(
            "Warning: no label/realized return column found. "
            "Continuing because Qlib portfolio backtest uses only prediction scores."
        )

    pred["datetime"] = pd.to_datetime(pred["datetime"])
    pred["instrument"] = pred["instrument"].astype(str)
    pred["score"] = pd.to_numeric(pred[score_col], errors="coerce")

    missing_score = pred["score"].isna().sum()
    if missing_score:
        raise ValueError(f"Prediction score contains {missing_score} NaN/non-numeric values.")

    normalized_cols = ["datetime", "instrument", "score"]
    if label_col is not None:
        pred["label"] = pd.to_numeric(pred[label_col], errors="coerce")
        normalized_cols.append("label")

    normalized = (
        pred[normalized_cols]
        .drop_duplicates(subset=["datetime", "instrument"], keep="last")
        .set_index(["datetime", "instrument"])
        .sort_index()
    )
    signal = normalized["score"].rename("score")
    return normalized, signal


def _resolve_universe(universe: str, provider_uri: Optional[str]) -> Tuple[str, str, str]:
    if universe not in UNIVERSE_SETTINGS:
        raise ValueError(f"Unsupported universe: {universe}. Expected one of {sorted(UNIVERSE_SETTINGS)}.")
    settings = UNIVERSE_SETTINGS[universe]
    resolved_provider = provider_uri or settings["provider_uri"]
    return (
        str(Path(resolved_provider).expanduser()),
        settings["region"],
        settings["benchmark"],
    )


def _default_output_dir(pred_path: Path, topk: int, drop: int) -> Path:
    seed = pred_path.stem.split("_", 1)[0]
    if not seed:
        seed = "unknown"
    return pred_path.parent / "backtest" / f"seed{seed}_top{topk}_drop{drop}"


def _make_metric_frame(report: pd.DataFrame) -> pd.DataFrame:
    from qlib.contrib.evaluate import risk_analysis

    calculate_table_metrics = _get_calculate_table_metrics()

    portfolio_return = report["return"].astype(float)
    benchmark_return = report["bench"].astype(float) if "bench" in report else None
    excess_return = portfolio_return - benchmark_return if benchmark_return is not None else None

    metric_sections: Dict[str, pd.Series] = {}

    qlib_port = risk_analysis(portfolio_return.dropna(), freq="day", mode="sum")["risk"]
    metric_sections["qlib_portfolio"] = qlib_port

    if benchmark_return is not None:
        metric_sections["qlib_benchmark"] = risk_analysis(benchmark_return.dropna(), freq="day", mode="sum")["risk"]
    if excess_return is not None:
        metric_sections["qlib_excess"] = risk_analysis(excess_return.dropna(), freq="day", mode="sum")["risk"]

    table_input = report[["return", "turnover"]].copy()
    table_metrics = calculate_table_metrics(table_input, period=None, name="project_portfolio")
    metric_sections["project_portfolio"] = table_metrics["project_portfolio"]

    if benchmark_return is not None:
        bench_input = pd.DataFrame({"return": benchmark_return, "turnover": 0.0}, index=report.index)
        bench_metrics = calculate_table_metrics(bench_input, period=None, name="project_benchmark")
        metric_sections["project_benchmark"] = bench_metrics["project_benchmark"]

    if excess_return is not None:
        excess_input = pd.DataFrame({"return": excess_return, "turnover": report["turnover"]}, index=report.index)
        excess_metrics = calculate_table_metrics(excess_input, period=None, name="project_excess")
        metric_sections["project_excess"] = excess_metrics["project_excess"]

    return pd.DataFrame(metric_sections)


def _save_outputs(
    output_dir: Path,
    report: pd.DataFrame,
    positions,
    metric_df: pd.DataFrame,
    signal: pd.Series,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    returns = pd.DataFrame(index=report.index)
    returns["return"] = report["return"]
    if "bench" in report:
        returns["benchmark"] = report["bench"]
        returns["excess_return"] = returns["return"] - returns["benchmark"]
    for optional_col in ["turnover", "total_turnover", "cost", "total_cost", "account", "value", "cash"]:
        if optional_col in report:
            returns[optional_col] = report[optional_col]

    returns.to_csv(output_dir / "portfolio_return.csv")
    report.to_csv(output_dir / "qlib_report.csv")
    metric_df.to_csv(output_dir / "portfolio_metric.csv")
    signal.to_frame("score").to_pickle(output_dir / "qlib_signal.pkl")

    if isinstance(positions, pd.DataFrame):
        positions.to_pickle(output_dir / "positions.pkl")
    else:
        pd.to_pickle(positions, output_dir / "positions.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qlib native portfolio backtest from PRISM-VQ stage2 prediction scores."
    )
    parser.add_argument("--pred_path", required=True, help="Path to stage2 *_best.pkl prediction file.")
    parser.add_argument("--universe", required=True, choices=sorted(UNIVERSE_SETTINGS), help="Market universe.")
    parser.add_argument("--qlib_provider_uri", default=None, help="Qlib data path. Defaults by universe.")
    parser.add_argument("--start_time", default="2022-01-01", help="Backtest start date.")
    parser.add_argument("--end_time", default="2024-12-31", help="Backtest end date.")
    parser.add_argument("--topk", type=int, default=30, help="Top-k holdings for Qlib TopkDropoutStrategy.")
    parser.add_argument("--drop", type=int, default=5, help="Number of holdings to replace each trading day.")
    parser.add_argument("--open_cost", type=float, default=0.0005, help="Open transaction cost.")
    parser.add_argument("--close_cost", type=float, default=0.0015, help="Close transaction cost.")
    parser.add_argument("--min_cost", type=float, default=0.0, help="Minimum transaction cost.")
    parser.add_argument("--account", type=float, default=100000000.0, help="Initial account value.")
    parser.add_argument("--benchmark", default=None, help="Override benchmark instrument.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to pred_path parent/backtest/seedX_topK_dropD.",
    )
    return parser.parse_args()


def main() -> None:
    import qlib
    from qlib.constant import REG_CN, REG_US
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy import TopkDropoutStrategy

    args = parse_args()
    pred_path = Path(args.pred_path).expanduser()
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    provider_uri, region, default_benchmark = _resolve_universe(args.universe, args.qlib_provider_uri)
    benchmark = args.benchmark or default_benchmark
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_output_dir(
        pred_path,
        topk=args.topk,
        drop=args.drop,
    )

    normalized_pred, signal = _normalize_prediction_frame(pred_path)
    print(f"Loaded prediction: {pred_path}")
    print(f"Signal rows: {len(signal):,}")
    print(f"Signal date range: {signal.index.get_level_values('datetime').min()} -> {signal.index.get_level_values('datetime').max()}")
    print(f"Signal instruments: {signal.index.get_level_values('instrument').nunique():,}")
    print(f"Output directory: {output_dir}")

    qlib_region = REG_US if region == "US" else REG_CN
    qlib.init(provider_uri=provider_uri, region=qlib_region)

    strategy = TopkDropoutStrategy(
        signal=signal,
        topk=args.topk,
        n_drop=args.drop,
    )

    report, positions = backtest_daily(
        start_time=args.start_time,
        end_time=args.end_time,
        strategy=strategy,
        account=args.account,
        benchmark=benchmark,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": None,
            "deal_price": "close",
            "open_cost": args.open_cost,
            "close_cost": args.close_cost,
            "min_cost": args.min_cost,
        },
    )

    metric_df = _make_metric_frame(report)
    _save_outputs(output_dir, report, positions, metric_df, signal)

    print("\nPortfolio metrics:")
    print(metric_df)
    print(f"\nSaved portfolio returns to {output_dir / 'portfolio_return.csv'}")
    print(f"Saved portfolio metrics to {output_dir / 'portfolio_metric.csv'}")
    print(f"Saved Qlib native report to {output_dir / 'qlib_report.csv'}")


if __name__ == "__main__":
    main()
