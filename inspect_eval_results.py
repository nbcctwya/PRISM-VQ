#!/usr/bin/env python
"""Validate Baseline Results Protocol v1.0 artifacts; exits nonzero on failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_results.config import MODEL_ID, RUNS, SEEDS, TEST
from baseline_results.metrics import PORTFOLIO_METRICS, RANKING_METRICS, portfolio_metrics, ranking_metrics
from generate_baseline_results import AGG_COLUMNS, BACKTEST, ENSEMBLE_COLUMNS, SEED_COLUMNS, STRATEGY, aggregate, load_prediction, make_ensemble


class Validator:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": bool(condition), "detail": detail})

    def attempt(self, name: str, fn) -> None:
        try:
            detail = fn()
            self.check(name, True, str(detail or "ok"))
        except Exception as exc:
            self.check(name, False, f"{type(exc).__name__}: {exc}")


def close(a, b, atol=1e-10) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-9, atol=atol, equal_nan=True))


def validate(out: Path, root: Path) -> dict:
    v = Validator()
    manifest_path = out / "metadata" / "manifest.json"
    v.check("manifest exists", manifest_path.exists(), str(manifest_path))
    if not manifest_path.exists():
        return finish(v)
    manifest = json.loads(manifest_path.read_text())
    for name, rel in manifest["files"].items():
        if "*" not in rel:
            v.check(f"manifest file: {name}", (out / rel).is_file(), rel)

    paths = {name: out / rel for name, rel in manifest["files"].items() if "*" not in rel}
    try:
        seed = pd.read_csv(paths["seed_metrics"])
        agg = pd.read_csv(paths["aggregate_metrics"])
        ens = pd.read_csv(paths["ensemble_metrics"], dtype={"seeds": str})
        seed_table = pd.read_csv(paths["seed_table"])
        ens_table = pd.read_csv(paths["ensemble_table"], dtype={"seeds": str})
        cfg = json.loads(paths["eval_config"].read_text())
    except Exception as exc:
        v.check("load required artifacts", False, str(exc))
        return finish(v)
    v.check("fixed CSV columns", list(seed) == SEED_COLUMNS and list(agg) == AGG_COLUMNS and list(ens) == ENSEMBLE_COLUMNS,
            "seed/aggregate/ensemble columns match schema")

    expected_seed = {(m, MODEL_ID, s) for m in RUNS for s in SEEDS}
    actual_seed = set(seed[["market", "model", "seed"]].itertuples(index=False, name=None))
    v.check("seed experiment completeness", actual_seed == expected_seed and len(seed) == len(expected_seed),
            f"expected={len(expected_seed)}, actual={len(seed)}, missing={sorted(expected_seed-actual_seed)}")
    v.check("seed primary key unique", not seed.duplicated(["market", "model", "seed"]).any(), "no duplicates")
    v.check("aggregate primary key unique", not agg.duplicated(["market", "model"]).any(), "no duplicates")
    v.check("ensemble primary key unique", not ens.duplicated(["market", "model", "ensemble_method"]).any(), "no duplicates")
    numeric_seed = seed[[*RANKING_METRICS, *PORTFOLIO_METRICS, "num_test_days"]].to_numpy(float)
    v.check("seed metrics finite", np.isfinite(numeric_seed).all(), "no allowed undefined exceptions in this run")
    v.check("seed metric bounds", (seed.IC.abs() <= 1).all() and (seed.RankIC.abs() <= 1).all()
            and (seed.STD >= 0).all() and (seed.MDD <= 0).all(), "IC/RankIC/STD/MDD bounds")
    expected_agg = aggregate(seed)
    v.check("aggregate recomputation", close(agg[AGG_COLUMNS[2:]], expected_agg[AGG_COLUMNS[2:]])
            and agg[["market", "model"]].equals(expected_agg[["market", "model"]]), "mean/std ddof=1")

    table_ok = True
    for _, row in agg.iterrows():
        shown = seed_table[(seed_table.market == row.market) & (seed_table.model == row.model)].iloc[0]
        for metric in (*RANKING_METRICS, *PORTFOLIO_METRICS):
            left, right = map(float, shown[metric].split(" ± "))
            table_ok &= abs(left - row[f"{metric}_mean"]) <= 5.1e-5 and abs(right - row[f"{metric}_std"]) <= 5.1e-5
    v.check("seed table derivation", table_ok, "four-decimal mean ± std")
    expected_ens = {(m, MODEL_ID, "avg_none") for m in RUNS}
    actual_ens = set(ens[["market", "model", "ensemble_method"]].itertuples(index=False, name=None))
    v.check("ensemble completeness", actual_ens == expected_ens and len(ens) == len(expected_ens), str(actual_ens))
    v.check("ensemble metrics finite", np.isfinite(ens[[*RANKING_METRICS, *PORTFOLIO_METRICS, "num_test_days"]].to_numpy(float)).all(), "all finite")
    v.check("ensemble display derivation", all(
        abs(float(ens_table.loc[i, metric]) - ens.loc[i, metric]) <= 5.1e-5
        for i in ens.index for metric in (*RANKING_METRICS, *PORTFOLIO_METRICS)), "four-decimal values")

    strategy_ok = all(cfg["strategy"].get(k) == val for k, val in (STRATEGY | {"freq": "day"}).items())
    bt_expected = BACKTEST | {"start_time": TEST[0], "end_time": TEST[1]}
    backtest_ok = all(cfg["backtest"].get(k) == val for k, val in bt_expected.items())
    v.check("fixed Qlib configuration", strategy_ok and backtest_ok, "all strategy/backtest constants explicit")
    v.check("signal timing", cfg["signal_alignment"] == {"signal_date": "t-1", "trade_date": "t",
            "qlib_internal_shift": 1, "adapter_manual_shift": False}, "Qlib strategy performs the only shift")

    for market in RUNS:
        curve_path = out / "curves" / "ensemble" / f"{market}_{MODEL_ID}.csv"
        score_path = out / "artifacts" / "ensemble" / f"{market}_{MODEL_ID}_avg_none.pkl"
        row = ens[(ens.market == market) & (ens.model == MODEL_ID)].iloc[0]
        def curve_check():
            c = pd.read_csv(curve_path, parse_dates=["datetime"])
            required = ["datetime", "daily_ret_gross", "cost", "daily_ret_net", "bench_ret", "nav", "bench_nav"]
            assert list(c) == required
            assert c.datetime.is_monotonic_increasing and c.datetime.is_unique
            assert np.isfinite(c.iloc[:, 1:].to_numpy(float)).all()
            assert close(c.daily_ret_net, c.daily_ret_gross - c.cost)
            assert close(c.nav, (1 + c.daily_ret_net).cumprod())
            assert close(c.bench_nav, (1 + c.bench_ret).cumprod())
            calculated = portfolio_metrics(c.daily_ret_net)
            assert all(np.isclose(calculated[k], row[k], rtol=1e-9, atol=1e-10) for k in (*PORTFOLIO_METRICS, "num_test_days"))
            return f"{len(c)} unique daily rows; curve and portfolio metrics reproduced"
        v.attempt(f"{market} ensemble curve", curve_check)

        def score_check():
            frames = [load_prediction(root / RUNS[market]["run_dir"] / f"{s}_best.pkl") for s in SEEDS]
            rebuilt = make_ensemble(frames)
            saved = pd.read_pickle(score_path)
            assert rebuilt.index.equals(saved.index) and close(rebuilt.score, saved.score) and close(rebuilt.label, saved.label)
            rank = ranking_metrics(saved)
            assert all(np.isclose(rank[k], row[k], rtol=1e-9, atol=1e-10) for k in RANKING_METRICS)
            return f"{len(saved)} aligned rows; raw-score average and ranking metrics reproduced"
        v.attempt(f"{market} ensemble score", score_check)

        def coverage_check():
            import qlib
            from qlib.constant import REG_CN, REG_US
            from qlib.data import D
            mc = RUNS[market]
            qlib.init(provider_uri=str(Path(mc["provider_uri"]).expanduser()), region=REG_CN if mc["region"] == "cn" else REG_US)
            cal = pd.DatetimeIndex(D.calendar(start_time=TEST[0], end_time=TEST[1], freq="day"))
            dates = pd.DatetimeIndex(pd.read_pickle(score_path).index.get_level_values("datetime").unique())
            assert len(cal.difference(dates)) == 0
            c = pd.read_csv(curve_path, parse_dates=["datetime"])
            assert c.datetime.min() == cal.min() and c.datetime.max() == cal.max() and len(c) == len(cal)
            return f"declared={TEST}; calendar={cal.min().date()}..{cal.max().date()}, {len(cal)} days"
        v.attempt(f"{market} test coverage", coverage_check)
    return finish(v)


def finish(v: Validator) -> dict:
    failures = sum(not c["passed"] for c in v.checks)
    return {"passed": failures == 0, "passes": len(v.checks) - failures, "failures": failures, "checks": v.checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.out if args.out.is_absolute() else root / args.out
    result = validate(out, root)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    (out / "diagnostics" / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "passes", "failures")}, ensure_ascii=False))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
