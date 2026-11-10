"""Part 1 → Part 2 handoff: feature transformation used by all later stages."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.paths import TEST_CSV, TRAIN_CSV

RAW_USECOLS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
]

NUM_COLS = [
    "amt",
    "city_pop",
    "running_total_amt",
    "running_total_transactions",
    "running_avg_amt",
    "amt_to_avg_ratio",
    "transactions_last_30d",
    "minutes_since_last_transaction",
    "is_first_transaction",
    "transactions_prev_10min",
    "transactions_prev_30min",
    "transactions_prev_60min",
    "cust_merch_distance_km",
    "merch_lat",
    "merch_long",
    "lat",
    "long",
    "age",
]

CAT_COLS = [
    "merchant",
    "category",
    "gender",
    "city",
    "state",
    "job",
    "day_of_week",
    "hour_of_day",
]

COLS_TO_DROP = [
    "is_fraud",
    "cc_num",
    "transaction_time",
    "time_since_last_transaction",
    "first",
    "last",
    "street",
    "zip",
    "trans_num",
    "unix_time",
    "dob",
    "total_prev_fraud",
    "prev_fraud_exists",
    "Unnamed: 0",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: transactions dataframe with a datetime ``transaction_time``.
    Output: same transactions plus past-looking features, ordered by time.
    """
    df = df.copy()
    df = df.sort_values(by=["cc_num", "transaction_time"]).reset_index(drop=True)

    df["running_total_amt"] = df.groupby("cc_num")["amt"].cumsum() - df["amt"]
    df["running_avg_amt"] = (
        (df.groupby("cc_num")["amt"].cumsum() - df["amt"])
        / df.groupby("cc_num").cumcount()
    ).fillna(0)
    df["running_total_transactions"] = df.groupby("cc_num").cumcount()
    df["time_since_last_transaction"] = df.groupby("cc_num")["transaction_time"].diff()

    df["transactions_last_30d"] = (
        df.set_index("transaction_time")
        .groupby("cc_num")["amt"]
        .transform(lambda x: x.rolling("30D").count() - 1)
        .reset_index(drop=True)
    ).fillna(0)

    df["amt_to_avg_ratio"] = (
        df["amt"] / df["running_avg_amt"].replace(0, np.nan)
    ).fillna(1)

    df["day_of_week"] = df["transaction_time"].dt.dayofweek
    df["hour_of_day"] = df["transaction_time"].dt.hour
    df["minutes_since_last_transaction"] = (
        df["time_since_last_transaction"].dt.total_seconds() / 60
    ).fillna(0)
    df["is_first_transaction"] = df["time_since_last_transaction"].isna().astype(int)

    # Past-looking previous fraud on the provided timeline (no current-row label)
    df["total_prev_fraud"] = (
        df.groupby("cc_num")["is_fraud"].cumsum() - df["is_fraud"]
    )
    df["prev_fraud_exist"] = (df["total_prev_fraud"] > 0).astype(int)

    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (
        df["transaction_time"].dt.year
        - df["dob"].dt.year
        - (
            (df["transaction_time"].dt.month < df["dob"].dt.month)
            | (
                (df["transaction_time"].dt.month == df["dob"].dt.month)
                & (df["transaction_time"].dt.day < df["dob"].dt.day)
            )
        ).astype(int)
    )

    lat1 = np.radians(df["lat"].to_numpy())
    lon1 = np.radians(df["long"].to_numpy())
    lat2 = np.radians(df["merch_lat"].to_numpy())
    lon2 = np.radians(df["merch_long"].to_numpy())
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["cust_merch_distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

    for t in [10, 30, 60]:
        df[f"transactions_prev_{t}min"] = (
            df.set_index("transaction_time")
            .groupby("cc_num")["cc_num"]
            .rolling(f"{t}min", closed="left")
            .count()
            .reset_index(drop=True)
            .fillna(0)
        )

    return df.sort_values("transaction_time").reset_index(drop=True)


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature matrix used by Part 2+."""
    out = df[NUM_COLS + CAT_COLS].copy()
    for col in CAT_COLS:
        out[col] = out[col].astype(str)
    return out


def make_xy(df: pd.DataFrame):
    """Return (X, y) using the project column contract."""
    drop_cols = [c for c in COLS_TO_DROP if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["is_fraud"].copy()
    return X, y


def _read_raw_csv(path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    cols = list(usecols) if usecols is not None else RAW_USECOLS
    # Keep only columns that exist (handles optional extras gracefully)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    use = [c for c in cols if c in header]
    df = pd.read_csv(path, usecols=use)
    df["transaction_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df.drop(columns=["trans_date_trans_time"])


def prepare_train_val_test(
    train_csv=None,
    test_csv=None,
    train_frac: float = 0.8,
    usecols: Iterable[str] | None = None,
):
    """
    Part 1→2 data prep (leakage-safe, history-continuous):

    1. Load train-file and test-file CSVs
    2. Concatenate, sort chronologically
    3. Run ``create_features`` once on the full timeline
    4. Split train-file rows into train (first ``train_frac``) / val (remainder)
    5. Test = all test-file rows (with prior train/val history available)

    Train-row features match split-then-transform when test is entirely after train.
    Val/test gain earlier card history. Model preprocessing remains train-only later.
    """
    train_csv = train_csv or TRAIN_CSV
    test_csv = test_csv or TEST_CSV

    df_train_file = _read_raw_csv(train_csv, usecols=usecols)
    df_test_file = _read_raw_csv(test_csv, usecols=usecols)
    df_train_file["data_source"] = "train_file"
    df_test_file["data_source"] = "test_file"

    combined = pd.concat([df_train_file, df_test_file], ignore_index=True)
    combined = combined.sort_values("transaction_time").reset_index(drop=True)
    combined = create_features(combined)

    train_file_mask = combined["data_source"] == "train_file"
    test_file_mask = combined["data_source"] == "test_file"

    train_file_rows = combined.loc[train_file_mask].sort_values(
        "transaction_time"
    ).reset_index(drop=True)
    split = int(len(train_file_rows) * train_frac)

    df_train = train_file_rows.iloc[:split].copy().reset_index(drop=True)
    df_val = train_file_rows.iloc[split:].copy().reset_index(drop=True)
    df_test = (
        combined.loc[test_file_mask]
        .sort_values("transaction_time")
        .reset_index(drop=True)
        .copy()
    )

    for frame in (df_train, df_val, df_test):
        if "data_source" in frame.columns:
            frame.drop(columns=["data_source"], inplace=True)

    return df_train, df_val, df_test
