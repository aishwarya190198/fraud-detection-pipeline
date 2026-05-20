"""
kafka_consumer.py
-----------------
Consumes transactions from Kafka topic in real time,
scores each one using the trained fraud model,
and flags suspicious transactions immediately.

This mirrors the event intelligence layer in Qinfinite —
consuming telemetry streams and making real-time decisions.

Prerequisites:
    pip install kafka-python
    Kafka running on localhost:9092
    Model trained: python src/train_model.py

Usage:
    python src/kafka_consumer.py
"""

import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

KAFKA_TOPIC  = "transactions"
KAFKA_BROKER = "localhost:9092"
THRESHOLD    = 0.4   # lower = catch more fraud (higher recall)


def load_model():
    with open("models/fraud_model.pkl", "rb") as f:
        return pickle.load(f)


def score_transaction(txn: dict, model_data: dict) -> dict:
    """Score a single transaction — same logic as batch pipeline."""
    model    = model_data["model"]
    scaler   = model_data["scaler"]
    features = model_data["features"]

    # Build feature vector (simplified for real-time — no customer history)
    row = {
        "amount":              txn.get("amount", 0),
        "hour_of_day":         txn.get("hour_of_day", 12),
        "day_of_week":         txn.get("day_of_week", 1),
        "is_online":           txn.get("is_online", 0),
        "is_foreign":          txn.get("is_foreign", 0),
        "is_weekend":          1 if txn.get("day_of_week", 1) >= 5 else 0,
        "is_night":            1 if txn.get("hour_of_day", 12) <= 4 else 0,
        "is_business_hour":    1 if 9 <= txn.get("hour_of_day", 12) <= 17 else 0,
        "cust_mean_amount":    500,   # default baseline for new customers
        "amount_zscore":       (txn.get("amount", 0) - 500) / 400,
        "hours_since_last_txn": 6,
        "txn_count_24h":       1,
        "merchant_risk_score": _merchant_risk(txn.get("merchant_type", "")),
        "amount_log":          np.log1p(txn.get("amount", 0)),
        "is_high_amount":      1 if txn.get("amount", 0) > 1500 else 0,
    }

    # Add merchant dummies
    merchants = [
        "ATM_Withdrawal", "Clothing", "Electronics", "Online_Shopping",
        "Petrol_Station", "Pharmacy", "Restaurant", "Supermarket",
        "Travel_Booking", "Utility_Bill"
    ]
    for m in merchants:
        row[f"merch_{m}"] = 1 if txn.get("merchant_type") == m else 0

    df = pd.DataFrame([row])

    # Align columns to training features
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    if scaler:
        df = pd.DataFrame(scaler.transform(df), columns=features)

    prob  = model.predict_proba(df)[0][1]
    label = 1 if prob >= THRESHOLD else 0

    return {
        "transaction_id": txn.get("transaction_id"),
        "customer_id":    txn.get("customer_id"),
        "amount":         txn.get("amount"),
        "merchant":       txn.get("merchant_type"),
        "fraud_prob":     round(float(prob), 4),
        "is_fraud":       label,
        "scored_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "alert":          "🔴 FRAUD ALERT" if label == 1 else "✅ CLEAR"
    }


def _merchant_risk(merchant):
    risk = {
        "Online_Shopping": 0.8, "Electronics": 0.7,
        "ATM_Withdrawal": 0.75, "Travel_Booking": 0.65,
        "Restaurant": 0.1, "Supermarket": 0.05,
        "Pharmacy": 0.05, "Petrol_Station": 0.1,
        "Clothing": 0.2,  "Utility_Bill": 0.05,
    }
    return risk.get(merchant, 0.3)


def consume_and_score():
    model_data = load_model()
    print(f"Model loaded: {model_data['model_name']}")
    print(f"Fraud threshold: {THRESHOLD}")
    print(f"Listening on topic: {KAFKA_TOPIC}")
    print("=" * 65)
    print(f"{'TXN ID':<14} {'Customer':<10} {'Amount':>8}  {'Prob':>6}  Alert")
    print("=" * 65)

    if not KAFKA_AVAILABLE:
        print("\nKafka not available — running batch scoring demo instead.")
        _batch_demo(model_data)
        return

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="fraud-scorer"
    )

    fraud_count = 0
    total_count = 0

    for msg in consumer:
        txn    = msg.value
        result = score_transaction(txn, model_data)
        total_count += 1

        if result["is_fraud"] == 1:
            fraud_count += 1

        print(f"{result['transaction_id']:<14} "
              f"{result['customer_id']:<10} "
              f"{result['amount']:>8.2f}  "
              f"{result['fraud_prob']:>6.3f}  "
              f"{result['alert']}")

        if total_count % 20 == 0:
            print(f"\n  Stats: {total_count} processed, "
                  f"{fraud_count} flagged ({fraud_count/total_count*100:.1f}%)\n")


def _batch_demo(model_data):
    """Demo mode — scores the test dataset without Kafka."""
    import os
    import sys
    sys.path.insert(0, ".")
    from src.data_generator import generate_transactions
    from src.feature_engineering import engineer_features, get_feature_columns

    if not os.path.exists("data/transactions.csv"):
        generate_transactions()

    df = pd.read_csv("data/transactions.csv").head(30)

    fraud_found = 0
    for _, row in df.iterrows():
        txn    = row.to_dict()
        result = score_transaction(txn, model_data)
        fraud_found += result["is_fraud"]
        print(f"{result['transaction_id']:<14} "
              f"{result['customer_id']:<10} "
              f"{result['amount']:>8.2f}  "
              f"{result['fraud_prob']:>6.3f}  "
              f"{result['alert']}")

    print(f"\nBatch demo complete: {fraud_found} / 30 flagged as fraud.")


if __name__ == "__main__":
    consume_and_score()
