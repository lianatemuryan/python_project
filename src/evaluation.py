"""
evaluation.py
-------------
Evaluates the trained classifier and saves results.

Saves:
  outputs/results/metrics.txt     — accuracy, precision, recall, confusion matrix
  outputs/results/predictions.csv — test set predictions vs actual labels
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay,
)


def evaluate_model(model, X_test, y_test, feature_names, le_target):
    """Evaluate model, print metrics, save results."""
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    # Make predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred, target_names=le_target.classes_)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nFull Classification Report:")
    print(report)

    # ------------------------------------------------------------------ #
    # Save metrics.txt
    # ------------------------------------------------------------------ #
    metrics_text = f"""
==============================================
  MACHINE LEARNING MODEL EVALUATION REPORT
==============================================

Task           : Binary Classification
Model          : Decision Tree Classifier (max_depth=5)
Target Column  : Learning_Mode (Online=1, Offline=0)

--- Metrics ---
Accuracy  : {accuracy:.4f}
Precision : {precision:.4f}
Recall    : {recall:.4f}

--- Confusion Matrix ---
              Predicted Offline    Predicted Online
Actual Offline       {cm[0][0]}                    {cm[0][1]}
Actual Online        {cm[1][0]}                    {cm[1][1]}

--- Full Classification Report ---
{report}

--- Explanation in Plain Words ---
* Machine Learning Task: Binary classification — predict whether a student
  learns Online or Offline based on study habits and performance metrics.

* Model Used: Decision Tree Classifier with max_depth=5.

* How well did it perform?
  Accuracy of {accuracy*100:.1f}% means the model correctly predicted the learning
  mode for about {int(accuracy*len(y_test))} out of {len(y_test)} test students.

* Metrics used: Accuracy, Precision, Recall, and Confusion Matrix.

* Did the model make many mistakes?
  The confusion matrix shows {cm[0][1] + cm[1][0]} total misclassifications out of
  {len(y_test)} test samples.

* What could improve the model?
  - Collecting more features (e.g., student age, location, device used)
  - Trying Random Forest or Logistic Regression for comparison
  - Hyperparameter tuning (e.g., varying max_depth, min_samples_split)
  - Feature engineering (e.g., interaction terms between Focus and Study_Hours)
==============================================
"""
    with open("outputs/results/metrics.txt", "w") as f:
        f.write(metrics_text)
    print("\nSaved: outputs/results/metrics.txt")

    # ------------------------------------------------------------------ #
    # Save predictions.csv
    # ------------------------------------------------------------------ #
    # NumPy task: use numpy to compute correct/incorrect flag
    correct_flag = np.where(y_pred == y_test.values, "Correct", "Incorrect")

    preds_df = pd.DataFrame({
        "Actual_Encoded":      y_test.values,
        "Predicted_Encoded":   y_pred,
        "Actual_Label":        le_target.inverse_transform(y_test.values),
        "Predicted_Label":     le_target.inverse_transform(y_pred),
        "Result":              correct_flag,
    })
    preds_df.to_csv("outputs/results/predictions.csv", index=False)
    print("Saved: outputs/results/predictions.csv")

    # ------------------------------------------------------------------ #
    # Save confusion matrix plot (bonus Matplotlib plot)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_target.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Decision Tree Classifier", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/plots/plot_6_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: outputs/plots/plot_6_confusion_matrix.png")

    return y_pred


if __name__ == "__main__":
    from preprocessing import preprocess
    from model import train_model
    X_train, X_test, y_train, y_test, features, le = preprocess("data/dataset.csv")
    mdl = train_model(X_train, y_train)
    evaluate_model(mdl, X_test, y_test, features, le)
