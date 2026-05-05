"""
data_exploration.py
-------------------
Dataset: Online vs Offline Learning Dataset (online_vs_offline_learning_dataset.csv)
Source: Provided by the student / course instructor.

Column descriptions:
  - Learning_Mode    : The mode of learning — "Online" or "Offline" (TARGET)
  - Subject          : Subject studied (e.g., English, Math, Science)
  - Study_Hours      : Daily study hours (float)
  - Retention_Score  : Memory retention score 0-100 (int)
  - Focus_Level      : Self-reported focus level 0-100 (int)
  - Exam_Score       : Final exam score 0-100 (int)

Machine Learning Task: CLASSIFICATION
Target column: Learning_Mode (predict whether a student studies Online or Offline)
"""

import pandas as pd
import numpy as np

def explore_data(filepath: str) -> pd.DataFrame:
    # 1. Load the dataset
    df = pd.read_csv(filepath)
    print("=" * 50)
    print("DATA EXPLORATION")
    print("=" * 50)

    # 2. Display first rows
    print("\n-- First 5 rows --")
    print(df.head())

    # 3. Dataset info
    print("\n-- Dataset Info --")
    df.info()

    # 4. Summary statistics
    print("\n-- Summary Statistics --")
    print(df.describe())

    # 5. Check for missing values
    print("\n-- Missing Values --")
    print(df.isnull().sum())

    # 6. Value counts for the target column
    print("\n-- Target Column Distribution (Learning_Mode) --")
    print(df["Learning_Mode"].value_counts())

    # 7. Unique subjects
    print("\n-- Unique Subjects --")
    print(df["Subject"].unique())

    # 8. Group by Learning_Mode and compute mean Exam_Score
    print("\n-- Mean Exam Score by Learning Mode --")
    print(df.groupby("Learning_Mode")["Exam_Score"].mean())

    # 9. Sort by Exam_Score descending
    print("\n-- Top 5 rows by Exam Score --")
    print(df.sort_values("Exam_Score", ascending=False).head())

    # 10. Filter rows where Study_Hours > 6
    print("\n-- Students who study more than 6 hours --")
    high_study = df[df["Study_Hours"] > 6]
    print(f"Count: {len(high_study)}")

    # NumPy task 1: mean study hours using numpy
    print("\n-- NumPy: Mean Study Hours --")
    print(np.mean(df["Study_Hours"].values))

    # NumPy task 2: standard deviation of Exam_Score
    print("\n-- NumPy: Std Dev of Exam Score --")
    print(np.std(df["Exam_Score"].values))

    return df


if __name__ == "__main__":
    df = explore_data("data/dataset.csv")
