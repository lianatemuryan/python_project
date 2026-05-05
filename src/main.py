"""
main.py
-------
Full pipeline runner. Run from the project root:

    python src/main.py

Steps:
  1. Data Exploration
  2. Preprocessing
  3. Visualization
  4. Model Training
  5. Evaluation
"""

import os
import sys

# Allow imports from src/ directory when running as `python src/main.py`
sys.path.insert(0, os.path.dirname(__file__))

from data_exploration import explore_data
from preprocessing    import preprocess
from visualization    import create_visualizations
from model            import train_model
from evaluation       import evaluate_model


DATASET_PATH = "data/dataset.csv"


def main():
    print("\n" + "=" * 60)
    print("  ONLINE vs OFFLINE LEARNING — ML PIPELINE")
    print("=" * 60)

    # Step 1: Explore
    print("\n[1/5] DATA EXPLORATION")
    explore_data(DATASET_PATH)

    # Step 2: Preprocess
    print("\n[2/5] PREPROCESSING")
    X_train, X_test, y_train, y_test, feature_cols, le_target = preprocess(DATASET_PATH)

    # Step 3: Visualize
    print("\n[3/5] VISUALIZATION")
    create_visualizations(DATASET_PATH)

    # Step 4: Train
    print("\n[4/5] MODEL TRAINING")
    model = train_model(X_train, y_train)

    # Step 5: Evaluate
    print("\n[5/5] EVALUATION")
    evaluate_model(model, X_test, y_test, feature_cols, le_target)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("  Plots   → outputs/plots/")
    print("  Results → outputs/results/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
