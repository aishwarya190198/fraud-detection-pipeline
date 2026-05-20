"""
kafka_producer.py
-----------------
Simulates a real-time bank transaction stream by sending
transactions to a Kafka topic one by one.

This represents the data ingestion layer of the Qinfinite
AIOps pipeline — live events flowing into the system.

Prerequisites:
    pip install kafka-python
    Kafka running locally on localhost:9092
    Create topic: kafka-topics.sh --create --topic transactions --bootstrap-server localhost:9092

Usage:
    python src/kafka_producer.py
"""

import json
import time
import random
import pandas as pd
from datetime import datetime

try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("kafka-python not installed. Run: pip install kafka-python")
    print("Running in SIMULATION MODE (printing to console instead)\n")

KAFKA_TOPIC   = "transactions"
KAFKA_BROKER  = "localhost:9092"
SEND_INTERVAL = 0.5  # seconds between transactions


def get_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3
    )


def stream_transactions(limit=100):
    """
    Reads from the generated dataset and streams
    transactions to Kafka (or prints if Kafka unavailable).
    """
    import os
    if not os.path.exists("data/transactions.csv"):
        print("Dataset not found. Run: python src/data_generator.py first.")
        return

    df = pd.read_csv("data/transactions.csv").head(limit)

    producer = get_producer() if KAFKA_AVAILABLE else None

    print(f"Streaming {limit} transactions to topic '{KAFKA_TOPIC}'...")
    print("-" * 55)
    print(f"{'#':<5} {'TXN ID':<14} {'Amount':>8}  {'Merchant':<18} {'Label'}")
    print("-" * 55)

    for i, row in df.iterrows():
        txn = {
            "transaction_id": row["transaction_id"],
            "customer_id":    row["customer_id"],
            "transaction_dt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "amount":         float(row["amount"]),
            "merchant_type":  row["merchant_type"],
            "country":        row["country"],
            "hour_of_day":    int(row["hour_of_day"]),
            "day_of_week":    int(row["day_of_week"]),
            "is_online":      int(row["is_online"]),
            "is_foreign":     int(row["is_foreign"]),
        }

        label = "🔴 FRAUD" if row["is_fraud"] == 1 else "✅ LEGIT"

        if producer:
            producer.send(KAFKA_TOPIC, value=txn)
            status = f"→ Kafka [{KAFKA_TOPIC}]"
        else:
            status = "→ SIMULATED"

        print(f"{i:<5} {row['transaction_id']:<14} {row['amount']:>8.2f}  "
              f"{row['merchant_type']:<18} {label}  {status}")

        time.sleep(SEND_INTERVAL)

    if producer:
        producer.flush()
        producer.close()
        print(f"\nDone. {limit} transactions sent to Kafka.")
    else:
        print(f"\nDone. {limit} transactions simulated.")


if __name__ == "__main__":
    stream_transactions(limit=50)
