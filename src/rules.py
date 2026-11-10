"""Part 1 rule candidates used by Parts 3 and 4."""

from __future__ import annotations

import pandas as pd

AMT_CANDIDATES = [150, 200, 220, 250, 280, 300]
MINUTES_BETWEEN_TRANSACTION = [20, 30, 40, 50, 60]
NUM_TRANSACTION_AT_23 = [2, 3, 4, 5]
NUM_TRANSACTION_60_MIN = [2, 3, 4, 5]

HIGH_RISK_CATEGORIES = ["shopping_net", "misc_net", "grocery_pos"]


def build_rule_strategies(df: pd.DataFrame, amt, minutes, tx_23, tx_60) -> dict:
    """Build high-precision / balanced / high-recall rule masks."""
    rule2 = (df["amt"] >= amt) & df["category"].isin(HIGH_RISK_CATEGORIES)
    rule3 = (
        (df["time_since_last_transaction"] <= pd.Timedelta(minutes=minutes))
        & (df["amt"] >= amt)
    )
    rule5 = df["transactions_prev_60min"] >= tx_60
    rule7 = (df["transactions_prev_60min"] >= tx_23) & (df["hour_of_day"] == 23)
    rule8 = df["prev_fraud_exist"] == 1

    high_precision = (
        (rule2 & rule7) | (rule3 & rule7) | (rule2 & rule5) | (rule3 & rule5)
    )
    balanced = high_precision | (rule2 & rule3)
    high_recall = balanced | (rule2 & rule8)

    return {
        "high_precision": high_precision,
        "balanced": balanced,
        "high_recall": high_recall,
    }


def apply_rule_config(df: pd.DataFrame, config: dict) -> pd.Series:
    """Apply a frozen rule configuration from strategy JSON."""
    strategies = build_rule_strategies(
        df,
        amt=config["amt_threshold"],
        minutes=config["minutes_threshold"],
        tx_23=config["transactions_23_threshold"],
        tx_60=config["transactions_60min_threshold"],
    )
    return strategies[config["strategy"]]
