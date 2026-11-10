"""
Full-timeline features → optimize at FP=$30 on validation → freeze →
test once → robustness under a small FP-cost grid (no re-tune, no retrain).
"""
from __future__ import annotations

import gc
import json
import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\asafk\Fraud Strategy Project")
sys.path.insert(0, str(ROOT))

from src.paths import ARTIFACTS_DIR, MODELS_DIR, STRATEGY_DIR, XGB_MODEL_PATH
from src.transform import prepare_model_frame, prepare_train_val_test

OUT_DIR = ARTIFACTS_DIR
STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

OPT_FP_COST = 30
ROBUSTNESS_FP_COSTS = [10, 30, 50, 100]

MODEL_THRESHOLDS = np.linspace(0.001, 0.999, 999)
HIGH_THRESHOLDS = np.linspace(0.02, 0.30, 281)
LOW_THRESHOLDS = np.linspace(0.001, 0.10, 100)

AMT_CANDIDATES = [150, 200, 220, 250, 280, 300]
NUM_TRANSACTION_AT_23 = [2, 3, 4, 5]
MINUTES_BETWEEN_TRANSACTION = [20, 30, 40, 50, 60]
NUM_TRANSACTION_60_MIN = [2, 3, 4, 5]


def log(msg: str) -> None:
    print(msg, flush=True)


def build_rule_strategies(df, amt, minutes, tx_23, tx_60):
    rule2 = (df["amt"] >= amt) & df["category"].isin(
        ["shopping_net", "misc_net", "grocery_pos"]
    )
    rule3 = (
        (df["time_since_last_transaction"] <= pd.Timedelta(minutes=minutes))
        & (df["amt"] >= amt)
    )
    rule5 = df["transactions_prev_60min"] >= tx_60
    rule7 = (df["transactions_prev_60min"] >= tx_23) & (df["hour_of_day"] == 23)
    rule8 = df["prev_fraud_exist"] == 1

    high_precision = (rule2 & rule7) | (rule3 & rule7) | (rule2 & rule5) | (rule3 & rule5)
    balanced = high_precision | (rule2 & rule3)
    high_recall = balanced | (rule2 & rule8)
    return {
        "high_precision": high_precision,
        "balanced": balanced,
        "high_recall": high_recall,
    }


def confusion_from_mask(y_true, pred):
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int((pred & y_true).sum())
    fp = int((pred & ~y_true).sum())
    fn = int((~pred & y_true).sum())
    tn = int((~pred & ~y_true).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn, fp_cost, fn_cost):
    return {
        "total_cost": float(fp * fp_cost + fn_cost),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def score_pred(y_true, pred, amounts, fp_cost):
    tp, fp, fn, tn = confusion_from_mask(y_true, pred)
    pred_b = np.asarray(pred).astype(bool)
    y_b = np.asarray(y_true).astype(bool)
    fn_cost = float(np.asarray(amounts)[(~pred_b) & y_b].sum())
    return metrics_from_counts(tp, fp, fn, tn, fp_cost, fn_cost)


def main():
    log("Preparing full-timeline features, then train/val/test split...")
    df_train, df_val, df_test = prepare_train_val_test()
    log(f"Shapes: train={df_train.shape}, val={df_val.shape}, test={df_test.shape}")
    log(
        f"Time ranges: train {df_train['transaction_time'].min()} → {df_train['transaction_time'].max()} | "
        f"val {df_val['transaction_time'].min()} → {df_val['transaction_time'].max()} | "
        f"test {df_test['transaction_time'].min()} → {df_test['transaction_time'].max()}"
    )

    # Free train frame — not needed for threshold opt / test
    del df_train
    gc.collect()

    y_val = df_val["is_fraud"].to_numpy()
    y_test = df_test["is_fraud"].to_numpy()
    amt_val = df_val["amt"].to_numpy()
    amt_test = df_test["amt"].to_numpy()

    log("Loading existing XGBoost model (no retrain)...")
    model = joblib.load(XGB_MODEL_PATH)

    log("Predicting validation probabilities...")
    xgb_val_prob = model.predict_proba(prepare_model_frame(df_val))[:, 1].astype(np.float32)
    gc.collect()

    log("Predicting test probabilities...")
    xgb_test_prob = model.predict_proba(prepare_model_frame(df_test))[:, 1].astype(np.float32)
    gc.collect()

    log(f"Optimizing rules on validation at FP=${OPT_FP_COST}...")
    best_rule = None
    for amt, minutes, tx_23, tx_60 in product(
        AMT_CANDIDATES,
        MINUTES_BETWEEN_TRANSACTION,
        NUM_TRANSACTION_AT_23,
        NUM_TRANSACTION_60_MIN,
    ):
        strategies = build_rule_strategies(df_val, amt, minutes, tx_23, tx_60)
        for strategy_name, rule in strategies.items():
            metrics = score_pred(y_val, rule, amt_val, OPT_FP_COST)
            row = {
                **metrics,
                "strategy": strategy_name,
                "amt_threshold": amt,
                "minutes_threshold": minutes,
                "transactions_23_threshold": tx_23,
                "transactions_60min_threshold": tx_60,
            }
            if best_rule is None or row["total_cost"] < best_rule["total_cost"]:
                best_rule = row

    log(
        f"Best rules @${OPT_FP_COST}: {best_rule['strategy']} | "
        f"amt>={best_rule['amt_threshold']}, minutes<={best_rule['minutes_threshold']}, "
        f"tx23>={best_rule['transactions_23_threshold']}, "
        f"tx60>={best_rule['transactions_60min_threshold']} | "
        f"val_cost={best_rule['total_cost']:.2f}"
    )

    log("Selecting best high-recall rule config for hybrid branch...")
    best_high_recall = None
    for amt, minutes, tx_23, tx_60 in product(
        AMT_CANDIDATES,
        MINUTES_BETWEEN_TRANSACTION,
        NUM_TRANSACTION_AT_23,
        NUM_TRANSACTION_60_MIN,
    ):
        rule = build_rule_strategies(df_val, amt, minutes, tx_23, tx_60)["high_recall"]
        metrics = score_pred(y_val, rule, amt_val, OPT_FP_COST)
        row = {
            **metrics,
            "strategy": "high_recall",
            "amt_threshold": amt,
            "minutes_threshold": minutes,
            "transactions_23_threshold": tx_23,
            "transactions_60min_threshold": tx_60,
        }
        if best_high_recall is None or row["total_cost"] < best_high_recall["total_cost"]:
            best_high_recall = row

    high_recall_val = build_rule_strategies(
        df_val,
        best_high_recall["amt_threshold"],
        best_high_recall["minutes_threshold"],
        best_high_recall["transactions_23_threshold"],
        best_high_recall["transactions_60min_threshold"],
    )["high_recall"].to_numpy()

    log(f"Optimizing XGBoost threshold on validation at FP=${OPT_FP_COST}...")
    best_xgb = None
    for thr in MODEL_THRESHOLDS:
        metrics = score_pred(y_val, xgb_val_prob >= thr, amt_val, OPT_FP_COST)
        row = {**metrics, "threshold": float(thr)}
        if best_xgb is None or row["total_cost"] < best_xgb["total_cost"]:
            best_xgb = row
    log(
        f"Best XGB threshold @${OPT_FP_COST}: {best_xgb['threshold']:.3f} | "
        f"val_cost={best_xgb['total_cost']:.2f}"
    )

    log(f"Optimizing hybrid thresholds on validation at FP=${OPT_FP_COST} (fine grid)...")
    best_hybrid = None
    for high_thr in HIGH_THRESHOLDS:
        for low_thr in LOW_THRESHOLDS:
            if low_thr >= high_thr:
                continue
            pred = (xgb_val_prob >= high_thr) | (
                (xgb_val_prob >= low_thr) & high_recall_val
            )
            metrics = score_pred(y_val, pred, amt_val, OPT_FP_COST)
            row = {
                **metrics,
                "high_threshold": float(high_thr),
                "low_threshold": float(low_thr),
            }
            if best_hybrid is None or row["total_cost"] < best_hybrid["total_cost"]:
                best_hybrid = row
    log(
        f"Best hybrid @${OPT_FP_COST}: high={best_hybrid['high_threshold']:.3f}, "
        f"low={best_hybrid['low_threshold']:.3f} | val_cost={best_hybrid['total_cost']:.2f}"
    )

    log("Freezing strategy and scoring test predictions once...")
    rules_test = build_rule_strategies(
        df_test,
        best_rule["amt_threshold"],
        best_rule["minutes_threshold"],
        best_rule["transactions_23_threshold"],
        best_rule["transactions_60min_threshold"],
    )[best_rule["strategy"]].to_numpy()

    high_recall_test = build_rule_strategies(
        df_test,
        best_high_recall["amt_threshold"],
        best_high_recall["minutes_threshold"],
        best_high_recall["transactions_23_threshold"],
        best_high_recall["transactions_60min_threshold"],
    )["high_recall"].to_numpy()

    xgb_test_pred = xgb_test_prob >= best_xgb["threshold"]
    hybrid_test_pred = (xgb_test_prob >= best_hybrid["high_threshold"]) | (
        (xgb_test_prob >= best_hybrid["low_threshold"]) & high_recall_test
    )

    frozen_components = {}
    for name, pred in [
        ("Rules only", rules_test),
        ("XGBoost only", xgb_test_pred),
        ("Hybrid", hybrid_test_pred),
    ]:
        tp, fp, fn, tn = confusion_from_mask(y_test, pred)
        pred_b = np.asarray(pred).astype(bool)
        fn_cost = float(amt_test[(~pred_b) & (y_test == 1)].sum())
        frozen_components[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "fn_cost": fn_cost,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }

    log("Computing robustness across FP-cost grid (no re-optimization)...")
    test_rows = []
    for fp_cost in ROBUSTNESS_FP_COSTS:
        for strategy, comp in frozen_components.items():
            row = {
                "fp_cost": fp_cost,
                "strategy": strategy,
                "total_cost": comp["fp"] * fp_cost + comp["fn_cost"],
                "precision": comp["precision"],
                "recall": comp["recall"],
                "tp": comp["tp"],
                "fp": comp["fp"],
                "fn": comp["fn"],
                "tn": comp["tn"],
            }
            if strategy == "XGBoost only":
                row["threshold"] = best_xgb["threshold"]
            if strategy == "Hybrid":
                row["high_threshold"] = best_hybrid["high_threshold"]
                row["low_threshold"] = best_hybrid["low_threshold"]
            test_rows.append(row)

    test_df = pd.DataFrame(test_rows)
    pivot = (
        test_df.pivot(index="fp_cost", columns="strategy", values="total_cost")
        .reset_index()
    )
    pivot["hybrid_vs_xgb_savings"] = pivot["XGBoost only"] - pivot["Hybrid"]
    pivot["hybrid_vs_xgb_pct_reduction"] = (
        100 * pivot["hybrid_vs_xgb_savings"] / pivot["XGBoost only"]
    )
    pivot["winner"] = pivot[["Rules only", "XGBoost only", "Hybrid"]].idxmin(axis=1)

    pct = pivot["hybrid_vs_xgb_pct_reduction"]
    summary = {
        "method": "full_timeline_features__optimize_at_fp_30__test_robustness",
        "optimization_fp_cost": OPT_FP_COST,
        "robustness_fp_costs": ROBUSTNESS_FP_COSTS,
        "frozen_xgb_threshold": best_xgb["threshold"],
        "frozen_hybrid_high_threshold": best_hybrid["high_threshold"],
        "frozen_hybrid_low_threshold": best_hybrid["low_threshold"],
        "mean_pct_reduction": float(pct.mean()),
        "median_pct_reduction": float(pct.median()),
        "min_pct_reduction": float(pct.min()),
        "max_pct_reduction": float(pct.max()),
        "std_pct_reduction": float(pct.std(ddof=0)),
        "hybrid_wins": int((pivot["winner"] == "Hybrid").sum()),
        "n_costs": int(len(pivot)),
        "baseline_30_pct_reduction": float(
            pivot.loc[pivot["fp_cost"] == 30, "hybrid_vs_xgb_pct_reduction"].iloc[0]
        ),
    }

    frozen_strategy = {
        "description": (
            "Features built on the full chronological timeline; strategy optimized on "
            "validation at FP=$30 only; evaluated for robustness on test under a small "
            "FP-cost grid without re-tuning. Existing XGBoost model reused (no retrain)."
        ),
        "model": {
            "name": "XGBoost",
            "path": "artifacts/models/best_xgb_model.joblib",
            "retrained_for_sensitivity": False,
        },
        "feature_prep": "prepare_train_val_test (full timeline then split)",
        "optimization_fp_cost": OPT_FP_COST,
        "baseline_fp_cost": OPT_FP_COST,
        "robustness_fp_costs": ROBUSTNESS_FP_COSTS,
        "fn_cost": "missed fraudulent transaction amount",
        "frozen": {
            "rules": {
                "strategy": best_rule["strategy"],
                "amt_threshold": int(best_rule["amt_threshold"]),
                "minutes_threshold": int(best_rule["minutes_threshold"]),
                "transactions_23_threshold": int(best_rule["transactions_23_threshold"]),
                "transactions_60min_threshold": int(
                    best_rule["transactions_60min_threshold"]
                ),
            },
            "xgb_threshold": float(best_xgb["threshold"]),
            "hybrid": {
                "high_threshold": float(best_hybrid["high_threshold"]),
                "low_threshold": float(best_hybrid["low_threshold"]),
                "rules": {
                    "strategy": "high_recall",
                    "amt_threshold": int(best_high_recall["amt_threshold"]),
                    "minutes_threshold": int(best_high_recall["minutes_threshold"]),
                    "transactions_23_threshold": int(
                        best_high_recall["transactions_23_threshold"]
                    ),
                    "transactions_60min_threshold": int(
                        best_high_recall["transactions_60min_threshold"]
                    ),
                },
            },
        },
        "test_robustness_summary": {
            "mean_pct_reduction_hybrid_vs_xgb": summary["mean_pct_reduction"],
            "median_pct_reduction_hybrid_vs_xgb": summary["median_pct_reduction"],
            "min_pct_reduction_hybrid_vs_xgb": summary["min_pct_reduction"],
            "max_pct_reduction_hybrid_vs_xgb": summary["max_pct_reduction"],
            "std_pct_reduction_hybrid_vs_xgb": summary["std_pct_reduction"],
            "baseline_30_pct_reduction_hybrid_vs_xgb": summary[
                "baseline_30_pct_reduction"
            ],
            "hybrid_wins": summary["hybrid_wins"],
            "n_costs": summary["n_costs"],
        },
    }

    pd.DataFrame([best_rule]).to_csv(OUT_DIR / "best_rules_by_fp_cost.csv", index=False)
    pd.DataFrame([best_xgb]).to_csv(
        OUT_DIR / "best_xgb_threshold_by_fp_cost.csv", index=False
    )
    pd.DataFrame([best_hybrid]).to_csv(OUT_DIR / "best_hybrid_by_fp_cost.csv", index=False)
    test_df.to_csv(OUT_DIR / "test_results_by_fp_cost.csv", index=False)
    pivot.to_csv(OUT_DIR / "fp_cost_sensitivity.csv", index=False)
    (OUT_DIR / "fp_cost_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (STRATEGY_DIR / "frozen_strategy.json").write_text(
        json.dumps(frozen_strategy, indent=2), encoding="utf-8"
    )

    log("\n=== Frozen @$30 thresholds ===")
    log(f"XGB threshold: {best_xgb['threshold']:.3f}")
    log(
        f"Hybrid: high={best_hybrid['high_threshold']:.3f}, "
        f"low={best_hybrid['low_threshold']:.3f}"
    )
    log("\n=== Robustness table ===")
    log(pivot.to_string(index=False))
    log("\n=== Summary ===")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
