"""
train_model.py
--------------
Trains and compares fraud detection models.
Handles class imbalance, evaluates with AUC-ROC and Precision-Recall.
Saves the best model for use in the streaming pipeline.

Usage:
    python src/train_model.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import sys
sys.path.insert(0, ".")
from src.data_generator import generate_transactions
from src.feature_engineering import engineer_features, get_feature_columns


def load_data():
    if not os.path.exists("data/transactions.csv"):
        generate_transactions()
    return pd.read_csv("data/transactions.csv")


def train_and_evaluate():
    print("=" * 60)
    print("  FRAUD DETECTION — MODEL TRAINING")
    print("=" * 60)

    # ── Load & engineer features ─────────────────────────────
    df_raw  = load_data()
    df      = engineer_features(df_raw)
    features = get_feature_columns(df)

    X = df[features]
    y = df["is_fraud"]

    print(f"\nDataset shape : {X.shape}")
    print(f"Fraud rate    : {y.mean()*100:.1f}%")

    # ── Train / test split (stratified) ──────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Scale for Logistic Regression ────────────────────────
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Class weights (handle imbalance without SMOTE) ────────
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw      = {0: weights[0], 1: weights[1]}
    print(f"Class weights : {cw}")

    # ── Define models ─────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=4, random_state=42
        ),
    }

    results   = {}
    trained   = {}

    print("\n" + "-" * 60)
    print(f"{'Model':<25} {'AUC-ROC':>10} {'Avg Precision':>15} {'F1 (fraud)':>12}")
    print("-" * 60)

    for name, model in models.items():
        X_tr = X_train_sc if name == "Logistic Regression" else X_train
        X_te = X_test_sc  if name == "Logistic Regression" else X_test

        model.fit(X_tr, y_train)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.4).astype(int)   # lower threshold for fraud

        auc  = roc_auc_score(y_test, y_prob)
        ap   = average_precision_score(y_test, y_prob)
        rep  = classification_report(y_test, y_pred, output_dict=True)
        f1   = rep.get("1", {}).get("f1-score", 0)

        results[name] = {
            "auc": auc, "avg_precision": ap, "f1_fraud": f1,
            "y_prob": y_prob, "y_pred": y_pred,
            "report": classification_report(y_test, y_pred)
        }
        trained[name] = (model, scaler if name == "Logistic Regression" else None)

        print(f"{name:<25} {auc:>10.4f} {ap:>15.4f} {f1:>12.4f}")

    print("-" * 60)

    # ── Best model ────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\nBest model: {best_name}  (AUC = {results[best_name]['auc']:.4f})")

    # ── Detailed report for best model ───────────────────────
    print(f"\nClassification Report — {best_name}:")
    print(results[best_name]["report"])

    # ── Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(y_test, results[best_name]["y_pred"])
    print("Confusion Matrix:")
    print(f"  True Negative  (Legit correctly cleared) : {cm[0][0]}")
    print(f"  False Positive (Legit flagged as fraud)  : {cm[0][1]}")
    print(f"  False Negative (Fraud missed)            : {cm[1][0]}")
    print(f"  True Positive  (Fraud correctly caught)  : {cm[1][1]}")

    # ── Feature importance (Random Forest) ───────────────────
    rf_model = trained["Random Forest"][0]
    fi = pd.DataFrame({
        "feature":   features,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False).head(15)

    # ── Save plots ────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    _plot_results(results, fi, y_test, cm, best_name)

    # ── Save best model ───────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    best_model, best_scaler = trained[best_name]
    with open("models/fraud_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "scaler": best_scaler,
                     "features": features, "model_name": best_name}, f)
    print("\nModel saved to: models/fraud_model.pkl")

    return results, features


def _plot_results(results, fi, y_test, cm, best_name):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Fraud Detection — Model Evaluation", fontsize=16, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = {"Logistic Regression": "#2196F3",
              "Random Forest":       "#4CAF50",
              "Gradient Boosting":   "#FF9800"}

    # 1. ROC curves
    ax1 = fig.add_subplot(gs[0, 0])
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax1.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=colors[name], lw=2)
    ax1.plot([0,1],[0,1],"k--", lw=1)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves"); ax1.legend(fontsize=8)

    # 2. Precision-Recall curves
    ax2 = fig.add_subplot(gs[0, 1])
    for name, res in results.items():
        p, r, _ = precision_recall_curve(y_test, res["y_prob"])
        ax2.plot(r, p, label=f"{name} (AP={res['avg_precision']:.3f})", color=colors[name], lw=2)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves"); ax2.legend(fontsize=8)

    # 3. Model comparison bar
    ax3 = fig.add_subplot(gs[0, 2])
    names = list(results.keys())
    aucs  = [results[n]["auc"] for n in names]
    bars  = ax3.bar(range(len(names)), aucs, color=[colors[n] for n in names], width=0.5)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax3.set_ylabel("AUC-ROC Score"); ax3.set_title("Model Comparison")
    ax3.set_ylim(0.5, 1.0)
    for bar, auc in zip(bars, aucs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{auc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4. Confusion matrix
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4,
                xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
    ax4.set_title(f"Confusion Matrix\n({best_name})")
    ax4.set_ylabel("Actual"); ax4.set_xlabel("Predicted")

    # 5. Feature importance
    ax5 = fig.add_subplot(gs[1, 1:])
    bars2 = ax5.barh(fi["feature"][::-1], fi["importance"][::-1], color="#4CAF50", alpha=0.8)
    ax5.set_xlabel("Importance Score")
    ax5.set_title("Top 15 Feature Importances (Random Forest)")
    ax5.tick_params(axis="y", labelsize=8)

    plt.savefig("outputs/model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plots saved to: outputs/model_evaluation.png")


if __name__ == "__main__":
    train_and_evaluate()
