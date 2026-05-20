"""
run_pipeline.py
---------------
Runs the complete fraud detection pipeline end to end.

Steps:
    1. Generate synthetic transaction dataset
    2. Run EDA and save charts
    3. Engineer features
    4. Train and evaluate models
    5. Demo real-time scoring (without Kafka)

Usage:
    python run_pipeline.py
"""

import sys
import os
sys.path.insert(0, ".")


def main():
    print("\n" + "=" * 60)
    print("  REAL-TIME FRAUD DETECTION PIPELINE")
    print("  Built by: Aishwarya Purbuj")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[Step 1/4] Generating transaction dataset...")
    from src.data_generator import generate_transactions
    generate_transactions()

    # Step 2: EDA
    print("\n[Step 2/4] Running exploratory data analysis...")
    from src.eda import run_eda
    run_eda()

    # Step 3 & 4: Train model
    print("\n[Step 3/4] Training and evaluating models...")
    from src.train_model import train_and_evaluate
    train_and_evaluate()

    # Step 5: Demo scoring
    print("\n[Step 4/4] Demo: Real-time transaction scoring...")
    from src.kafka_consumer import _batch_demo, load_model
    model_data = load_model()
    _batch_demo(model_data)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print("\nOutputs:")
    print("  data/transactions.csv       — Generated dataset")
    print("  models/fraud_model.pkl      — Trained model")
    print("  outputs/eda_overview.png    — EDA charts")
    print("  outputs/eda_correlation.png — Correlation heatmap")
    print("  outputs/model_evaluation.png— Model comparison charts")
    print("\nTo run with real Kafka:")
    print("  Terminal 1: python src/kafka_producer.py")
    print("  Terminal 2: python src/kafka_consumer.py")


if __name__ == "__main__":
    main()
