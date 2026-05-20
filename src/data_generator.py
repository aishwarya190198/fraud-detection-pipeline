"""
data_generator.py
-----------------
Generates a realistic synthetic bank transaction dataset
with fraud patterns built in. Run this first.

Usage:
    python src/data_generator.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N_LEGIT  = 9700
N_FRAUD  = 300
N_TOTAL  = N_LEGIT + N_FRAUD

MERCHANTS = [
    "Supermarket", "Restaurant", "Online_Shopping", "Pharmacy",
    "Petrol_Station", "Electronics", "Clothing", "ATM_Withdrawal",
    "Utility_Bill", "Travel_Booking"
]

HIGH_RISK_MERCHANTS = ["Online_Shopping", "Electronics", "ATM_Withdrawal", "Travel_Booking"]

COUNTRIES = ["IN", "GB", "US", "DE", "IE", "NL", "AE", "SG"]


def random_datetime(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def generate_transactions():
    start_date = datetime(2024, 1, 1)
    end_date   = datetime(2024, 12, 31)

    records = []

    # ── Legitimate transactions ───────────────────────────────
    for i in range(N_LEGIT):
        txn_dt    = random_datetime(start_date, end_date)
        merchant  = random.choice(MERCHANTS)
        amount    = round(np.random.lognormal(mean=4.0, sigma=0.8), 2)  # realistic spend
        amount    = min(amount, 3000)

        records.append({
            "transaction_id":   f"TXN{i:06d}",
            "customer_id":      f"CUST{random.randint(1000, 2000):04d}",
            "transaction_dt":   txn_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "amount":           amount,
            "merchant_type":    merchant,
            "country":          random.choices(COUNTRIES, weights=[60,10,8,5,5,5,4,3])[0],
            "hour_of_day":      txn_dt.hour,
            "day_of_week":      txn_dt.weekday(),
            "is_online":        1 if merchant in ["Online_Shopping", "Travel_Booking"] else 0,
            "is_foreign":       0,
            "is_fraud":         0
        })

    # ── Fraudulent transactions ───────────────────────────────
    for i in range(N_FRAUD):
        txn_dt   = random_datetime(start_date, end_date)
        merchant = random.choice(HIGH_RISK_MERCHANTS)

        # Fraud patterns: high amount, odd hours, foreign country
        pattern = random.choice(["high_amount", "odd_hour", "foreign", "velocity"])

        if pattern == "high_amount":
            amount = round(np.random.uniform(2000, 9000), 2)
            foreign = 0
            hour    = random.randint(8, 20)
        elif pattern == "odd_hour":
            amount  = round(np.random.uniform(500, 3000), 2)
            foreign = 0
            hour    = random.choice([0, 1, 2, 3, 4])
        elif pattern == "foreign":
            amount  = round(np.random.uniform(300, 5000), 2)
            foreign = 1
            hour    = random.randint(0, 23)
        else:  # velocity
            amount  = round(np.random.uniform(100, 800), 2)
            foreign = random.choice([0, 1])
            hour    = random.randint(0, 23)

        records.append({
            "transaction_id": f"TXN_F{i:06d}",
            "customer_id":    f"CUST{random.randint(1000, 2000):04d}",
            "transaction_dt": txn_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "amount":         amount,
            "merchant_type":  merchant,
            "country":        random.choice(["AE", "SG", "US", "GB"]) if foreign else "IN",
            "hour_of_day":    hour,
            "day_of_week":    txn_dt.weekday(),
            "is_online":      1,
            "is_foreign":     foreign,
            "is_fraud":       1
        })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)

    print(f"Dataset created: {len(df)} transactions")
    print(f"  Legitimate : {len(df[df.is_fraud==0])}")
    print(f"  Fraudulent : {len(df[df.is_fraud==1])}")
    print(f"  Fraud rate : {df.is_fraud.mean()*100:.1f}%")
    print(f"Saved to: data/transactions.csv")
    return df


if __name__ == "__main__":
    generate_transactions()
