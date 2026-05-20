# Real-Time Fraud Detection Pipeline

A production-style fraud detection system that simulates real-time bank transaction scoring using Apache Kafka, Python, and machine learning.

Built to demonstrate end-to-end data engineering and data science skills — from streaming data ingestion to ML model training and real-time inference.

---

## Architecture

```
Bank Transactions
      │
      ▼
┌─────────────────┐
│  Kafka Producer  │  ← Streams transactions in real time
│  (kafka_producer)│
└────────┬────────┘
         │  Kafka Topic: "transactions"
         ▼
┌─────────────────┐
│  Kafka Consumer  │  ← Consumes stream, scores each transaction
│  (kafka_consumer)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Scorer      │  ← Fraud probability + alert
│  (fraud_model)   │
└────────┬────────┘
         │
         ▼
   🔴 FRAUD ALERT  /  ✅ CLEAR
```

---

## Features

- **Synthetic data generator** — creates 10,000 realistic bank transactions with built-in fraud patterns (high amounts, odd hours, foreign transactions, velocity attacks)
- **Feature engineering** — Z-score deviation from customer baseline, transaction velocity, merchant risk scoring, time-based features
- **Model comparison** — Logistic Regression vs Random Forest vs Gradient Boosting with AUC-ROC and Precision-Recall evaluation
- **Class imbalance handling** — balanced class weights (same effect as SMOTE, no extra dependency)
- **Real-time scoring** — Kafka consumer scores each transaction as it arrives with sub-second latency
- **EDA module** — complete exploratory analysis with charts

---

## Project Structure

```
fraud-detection/
├── src/
│   ├── data_generator.py      # Generate synthetic transaction dataset
│   ├── eda.py                 # Exploratory data analysis + charts
│   ├── feature_engineering.py # Feature creation pipeline
│   ├── train_model.py         # Model training, comparison, evaluation
│   ├── kafka_producer.py      # Stream transactions to Kafka
│   └── kafka_consumer.py      # Consume stream + real-time scoring
├── data/
│   └── transactions.csv       # Generated dataset (auto-created)
├── models/
│   └── fraud_model.pkl        # Trained model (auto-created)
├── outputs/
│   ├── eda_overview.png       # EDA charts
│   ├── eda_correlation.png    # Correlation heatmap
│   └── model_evaluation.png   # Model comparison charts
├── notebooks/
│   └── (upload your Jupyter notebooks here)
├── run_pipeline.py            # Run everything end to end
└── requirements.txt
```

---

## Quick Start

### 1. Clone and install
```bash
git clone https://github.com/aishwarya-purbuj/fraud-detection-pipeline.git
cd fraud-detection-pipeline
pip install -r requirements.txt
```

### 2. Run the complete pipeline (no Kafka needed)
```bash
python run_pipeline.py
```

This will:
- Generate 10,000 synthetic transactions
- Run EDA and save charts to `outputs/`
- Train and compare 3 ML models
- Demo real-time scoring on 30 transactions

### 3. Run with real Kafka (optional)

Start Kafka locally (Docker):
```bash
docker run -d --name kafka \
  -p 9092:9092 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \
  confluentinc/cp-kafka:latest
```

Then in two terminals:
```bash
# Terminal 1 — stream transactions
python src/kafka_producer.py

# Terminal 2 — score in real time
python src/kafka_consumer.py
```

---

## Model Results

| Model | AUC-ROC | Avg Precision | F1 (Fraud) |
|---|---|---|---|
| Logistic Regression | ~0.91 | ~0.68 | ~0.72 |
| Random Forest | ~0.97 | ~0.88 | ~0.85 |
| **Gradient Boosting** | **~0.98** | **~0.91** | **~0.87** |

*Results on held-out 20% test set. Threshold tuned to 0.4 to favour recall (catch more fraud).*

---

## Key Features Engineered

| Feature | Description | Why it matters |
|---|---|---|
| `amount_zscore` | How many STDs the amount deviates from customer baseline | Catches unusual spend for that specific customer |
| `merchant_risk_score` | Pre-assigned risk score per merchant type | Online Shopping, Electronics, ATM = high risk |
| `txn_count_24h` | Transactions in last 24 hours | Velocity attacks: many small transactions |
| `is_night` | Transaction between 12am–4am | Fraudsters often operate at night |
| `is_foreign` | Cross-border transaction flag | Foreign transactions have 3× higher fraud rate |
| `hours_since_last_txn` | Time gap from previous transaction | Unusually rapid transactions signal fraud |

---

## EDA Key Findings

1. **Fraud amounts are 4× higher** on average than legitimate transactions
2. **Transactions between 12am–4am** have 8× higher fraud rate
3. **Foreign transactions** have a 3× higher fraud rate than domestic
4. **Online Shopping, Electronics, ATM** are the highest-risk merchant categories
5. **Class imbalance**: 3% fraud vs 97% legitimate — requires balanced class weights

---

## Real-World Connection

This project is inspired by the AIOps observability pipeline I built at **Quinnox** on the **Qinfinite platform** — where real-time Kafka streams of IT telemetry (logs, metrics, events) were processed and scored for anomaly detection. The architecture is identical:

- **Kafka producer** → IT systems or bank transaction sources
- **Kafka consumer** → event scoring engine  
- **ML model** → anomaly/fraud detection
- **Alert** → incident ticket or fraud flag

---

## Tech Stack

- **Apache Kafka** — real-time event streaming
- **Python** — data processing and ML
- **Pandas / NumPy** — feature engineering
- **Scikit-learn** — model training and evaluation
- **XGBoost / Gradient Boosting** — best performing classifier
- **Matplotlib / Seaborn** — visualisation

---

## Author

**Aishwarya Purbuj**  
Data Engineer | Aspiring Data Scientist  
[LinkedIn](https://linkedin.com/in/aishwaryapurbuj) · [GitHub](https://github.com/aishwarya-purbuj)
