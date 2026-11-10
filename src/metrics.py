"""Cost-sensitive metrics shared by Parts 3 and 4."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FP_COSTS = [10, 30, 50, 100]


def evaluate_rule(rule: pd.Series, df: pd.DataFrame) -> dict:
    tp = int(((rule) & (df["is_fraud"] == 1)).sum())
    fp = int(((rule) & (df["is_fraud"] == 0)).sum())
    fn = int(((~rule) & (df["is_fraud"] == 1)).sum())
    tn = int(((~rule) & (df["is_fraud"] == 0)).sum())
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "coverage": float(rule.mean()),
        "flagged": int(rule.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_rule_with_cost(rule: pd.Series, df: pd.DataFrame, fp_cost: float = 30) -> dict:
    metrics = evaluate_rule(rule, df)
    total_fn_cost = float(
        df.loc[(~rule) & (df["is_fraud"] == 1), "amt"].sum()
    )
    metrics["total_cost"] = metrics["fp"] * fp_cost + total_fn_cost
    metrics["fp_cost"] = fp_cost
    return metrics


def evaluate_predictions(y_true, predictions, amounts, fp_cost: float = 30) -> dict:
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions).astype(bool)
    amounts = np.asarray(amounts)

    tp = int(((predictions == 1) & (y_true == 1)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    missed_fraud_cost = float(amounts[(predictions == 0) & (y_true == 1)].sum())

    return {
        "fp_cost": fp_cost,
        "total_cost": fp * fp_cost + missed_fraud_cost,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_model_threshold(
    y_true, probabilities, amounts, threshold, fp_cost: float = 30
) -> dict:
    predictions = np.asarray(probabilities) >= threshold
    result = evaluate_predictions(y_true, predictions, amounts, fp_cost=fp_cost)
    result["threshold"] = float(threshold)
    return result
