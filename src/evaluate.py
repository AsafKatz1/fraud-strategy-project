"""Part 3 → Part 4 helpers: load frozen strategy and score test predictions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.metrics import evaluate_predictions
from src.paths import FROZEN_STRATEGY_PATH, XGB_MODEL_PATH
from src.rules import apply_rule_config
from src.transform import prepare_model_frame


def load_frozen_strategy(path: Path | None = None) -> dict:
    path = path or FROZEN_STRATEGY_PATH
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_xgb_model(path: Path | None = None):
    path = path or XGB_MODEL_PATH
    return joblib.load(path)


def hybrid_predict(probabilities, rules_mask, high_threshold, low_threshold):
    probabilities = np.asarray(probabilities)
    rules_mask = np.asarray(rules_mask).astype(bool)
    return (probabilities >= high_threshold) | (
        (probabilities >= low_threshold) & rules_mask
    )


def score_frozen_strategy(
    df_test: pd.DataFrame,
    y_test,
    strategy: dict | None = None,
    model=None,
    fp_cost: float | None = None,
) -> dict:
    """
    Score the frozen @$30 package on the test set.

    ``fp_cost`` only changes the cost accounting; predictions stay frozen.
    """
    strategy = strategy or load_frozen_strategy()
    model = model or load_xgb_model()
    fp_cost = strategy.get("baseline_fp_cost", 30) if fp_cost is None else fp_cost

    frozen = strategy["frozen"]
    rules_cfg = frozen["rules"]
    xgb_thr = frozen["xgb_threshold"]
    hybrid_cfg = frozen["hybrid"]

    X_test = prepare_model_frame(df_test)
    probs = model.predict_proba(X_test)[:, 1]
    amounts = df_test["amt"]

    rules_pred = apply_rule_config(df_test, rules_cfg)
    hybrid_rules = apply_rule_config(df_test, hybrid_cfg["rules"])
    xgb_pred = probs >= xgb_thr
    hybrid_pred = hybrid_predict(
        probs,
        hybrid_rules,
        hybrid_cfg["high_threshold"],
        hybrid_cfg["low_threshold"],
    )

    return {
        "fp_cost": fp_cost,
        "rules": evaluate_predictions(y_test, rules_pred, amounts, fp_cost=fp_cost),
        "xgboost": evaluate_predictions(y_test, xgb_pred, amounts, fp_cost=fp_cost),
        "hybrid": evaluate_predictions(y_test, hybrid_pred, amounts, fp_cost=fp_cost),
        "chosen": frozen,
    }
