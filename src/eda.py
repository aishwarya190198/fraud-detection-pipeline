"""
eda.py
------
Exploratory Data Analysis for the fraud detection dataset.
Run this to generate all EDA charts saved to outputs/

Usage:
    python src/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys
sys.path.insert(0, ".")

from src.data_generator import generate_transactions

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("outputs", exist_ok=True)


def run_eda():
    # ── Load data ────────────────────────────────────────────
    if not os.path.exists("data/transactions.csv"):
        generate_transactions()
    df = pd.read_csv("data/transactions.csv")

    print("=" * 55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 55)
    print(f"\nShape    : {df.shape}")
    print(f"Columns  : {list(df.columns)}")
    print(f"\nClass distribution:")
    print(df["is_fraud"].value_counts())
    print(f"\nFraud rate: {df['is_fraud'].mean()*100:.1f}%")
    print(f"\nAmount stats:")
    print(df.groupby("is_fraud")["amount"].describe().round(2))

    # ── Plot 1: Overview ─────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Fraud Detection — Exploratory Data Analysis", fontsize=14, fontweight="bold")

    # 1a. Class imbalance
    ax = axes[0, 0]
    counts = df["is_fraud"].value_counts()
    bars = ax.bar(["Legitimate", "Fraud"], counts.values,
                  color=["#4CAF50", "#F44336"], alpha=0.8, width=0.5)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{val:,}", ha="center", fontweight="bold")

    # 1b. Amount distribution
    ax = axes[0, 1]
    ax.hist(df[df.is_fraud==0]["amount"], bins=50, alpha=0.6,
            color="#4CAF50", label="Legitimate", density=True)
    ax.hist(df[df.is_fraud==1]["amount"], bins=50, alpha=0.6,
            color="#F44336", label="Fraud", density=True)
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Amount (₹)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(0, 10000)

    # 1c. Fraud by hour
    ax = axes[0, 2]
    fraud_by_hour = df.groupby("hour_of_day")["is_fraud"].mean() * 100
    ax.bar(fraud_by_hour.index, fraud_by_hour.values, color="#FF9800", alpha=0.8)
    ax.set_title("Fraud Rate by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Fraud Rate (%)")
    ax.axvspan(-0.5, 4.5, alpha=0.1, color="red", label="High-risk hours (0-4am)")
    ax.legend(fontsize=8)

    # 1d. Fraud by merchant
    ax = axes[1, 0]
    fraud_by_merch = (
        df.groupby("merchant_type")["is_fraud"].mean() * 100
    ).sort_values(ascending=True)
    colors_bar = ["#F44336" if v > 5 else "#4CAF50" for v in fraud_by_merch.values]
    ax.barh(fraud_by_merch.index, fraud_by_merch.values, color=colors_bar, alpha=0.8)
    ax.set_title("Fraud Rate by Merchant Type")
    ax.set_xlabel("Fraud Rate (%)")

    # 1e. Amount boxplot by fraud
    ax = axes[1, 1]
    df_plot = df[df["amount"] < 5000]
    data_legit = df_plot[df_plot.is_fraud==0]["amount"]
    data_fraud = df_plot[df_plot.is_fraud==1]["amount"]
    ax.boxplot([data_legit, data_fraud], labels=["Legitimate", "Fraud"],
               patch_artist=True,
               boxprops=dict(facecolor="#e3f2fd"),
               medianprops=dict(color="#F44336", linewidth=2))
    ax.set_title("Amount Distribution by Class")
    ax.set_ylabel("Amount (₹)")

    # 1f. Foreign transactions
    ax = axes[1, 2]
    foreign_fraud = df.groupby(["is_foreign", "is_fraud"]).size().unstack(fill_value=0)
    foreign_fraud.plot(kind="bar", ax=ax, color=["#4CAF50", "#F44336"],
                       alpha=0.8, rot=0, legend=True)
    ax.set_xticklabels(["Domestic", "Foreign"])
    ax.set_title("Domestic vs Foreign Transactions")
    ax.set_ylabel("Count")
    ax.legend(["Legitimate", "Fraud"])

    plt.tight_layout()
    plt.savefig("outputs/eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nEDA chart saved: outputs/eda_overview.png")

    # ── Plot 2: Correlation heatmap ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    num_cols = ["amount", "hour_of_day", "day_of_week",
                "is_online", "is_foreign", "is_fraud"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/eda_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Correlation chart saved: outputs/eda_correlation.png")

    # ── Key findings ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  KEY FINDINGS")
    print("=" * 55)
    avg_fraud  = df[df.is_fraud==1]["amount"].mean()
    avg_legit  = df[df.is_fraud==0]["amount"].mean()
    night_rate = df[df.hour_of_day <= 4]["is_fraud"].mean() * 100
    day_rate   = df[df.hour_of_day > 4]["is_fraud"].mean() * 100
    foreign_rate = df[df.is_foreign==1]["is_fraud"].mean() * 100

    print(f"1. Avg fraud amount   : ₹{avg_fraud:,.0f}  vs  ₹{avg_legit:,.0f} (legitimate)")
    print(f"2. Night fraud rate   : {night_rate:.1f}%  vs  {day_rate:.1f}% daytime")
    print(f"3. Foreign txn fraud  : {foreign_rate:.1f}% of foreign transactions are fraud")
    print(f"4. High-risk merchants: Online Shopping, Electronics, ATM Withdrawal")


if __name__ == "__main__":
    run_eda()
