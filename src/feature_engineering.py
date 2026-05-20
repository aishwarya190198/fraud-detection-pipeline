"""
feature_engineering.py
-----------------------
Transforms raw transaction data into ML-ready features.
Mirrors the kind of feature work done in Qinfinite AIOps pipelines.

Usage:
    from src.feature_engineering import engineer_features
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features from raw transaction data.
    These features are the core of what makes fraud detectable.
    """
    df = df.copy()
    df["transaction_dt"] = pd.to_datetime(df["transaction_dt"])
    df = df.sort_values(["customer_id", "transaction_dt"]).reset_index(drop=True)

    # ── Time-based features ──────────────────────────────────
    df["is_weekend"]       = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]         = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 4)).astype(int)
    df["is_business_hour"] = ((df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 17)).astype(int)

    # ── Customer spend baseline (Z-score deviation) ───────────
    # This is the same Z-score anomaly detection used in Qinfinite
    cust_stats = df.groupby("customer_id")["amount"].agg(
        cust_mean_amount="mean",
        cust_std_amount="std"
    ).reset_index()
    df = df.merge(cust_stats, on="customer_id", how="left")
    df["cust_std_amount"] = df["cust_std_amount"].fillna(1)
    df["amount_zscore"] = (
        (df["amount"] - df["cust_mean_amount"]) / df["cust_std_amount"]
    ).round(4)

    # ── Transaction velocity (last N transactions) ────────────
    df = df.sort_values(["customer_id", "transaction_dt"])
    df["prev_txn_dt"] = df.groupby("customer_id")["transaction_dt"].shift(1)
    df["hours_since_last_txn"] = (
        (df["transaction_dt"] - df["prev_txn_dt"]).dt.total_seconds() / 3600
    ).fillna(24).round(2)

    # Transactions in last 24 hours (velocity)
    df["txn_count_24h"] = (
        df.groupby("customer_id")
        .apply(lambda g: g["transaction_dt"]
               .apply(lambda t: ((t - g["transaction_dt"]) < pd.Timedelta("24h")).sum()))
        .reset_index(level=0, drop=True)
    )

    # ── Merchant risk score ───────────────────────────────────
    merchant_risk = {
        "Online_Shopping":  0.8,
        "Electronics":      0.7,
        "ATM_Withdrawal":   0.75,
        "Travel_Booking":   0.65,
        "Restaurant":       0.1,
        "Supermarket":      0.05,
        "Pharmacy":         0.05,
        "Petrol_Station":   0.1,
        "Clothing":         0.2,
        "Utility_Bill":     0.05,
    }
    df["merchant_risk_score"] = df["merchant_type"].map(merchant_risk).fillna(0.3)

    # ── Amount buckets ────────────────────────────────────────
    df["amount_log"] = np.log1p(df["amount"])
    df["is_high_amount"] = (df["amount"] > df["cust_mean_amount"] * 3).astype(int)

    # ── Encode merchant type ──────────────────────────────────
    df = pd.get_dummies(df, columns=["merchant_type"], prefix="merch", drop_first=False)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Returns the list of feature columns for model training."""
    exclude = [
        "transaction_id", "customer_id", "transaction_dt",
        "prev_txn_dt", "country", "is_fraud",
        "cust_mean_amount", "cust_std_amount"
    ]
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_generator import generate_transactions

    df_raw = generate_transactions()
    df_feat = engineer_features(df_raw)
    print("\nFeature Engineering complete.")
    print(f"Features created: {len(get_feature_columns(df_feat))}")
    print(get_feature_columns(df_feat))
