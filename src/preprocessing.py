"""
preprocessing.py
----------------
Cleans and prepares the dataset for machine learning.

Features: Study_Hours, Retention_Score, Focus_Level, Exam_Score, Subject (encoded)
Target:   Learning_Mode (encoded: Online=1, Offline=0)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess(filepath: str):
    """Returns X_train, X_test, y_train, y_test, feature names."""

    # Preprocessing task 1: load dataset
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")

    # Preprocessing task 2: handle missing values (fill numeric with median)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Pandas task 11: create a new column — Study Efficiency
    df["Study_Efficiency"] = df["Exam_Score"] / (df["Study_Hours"] + 1)

    # Pandas task 12: rename column to make it clearer
    df.rename(columns={"Retention_Score": "Memory_Retention"}, inplace=True)

    # Preprocessing task 3: encode categorical Subject column
    le_subject = LabelEncoder()
    df["Subject_Encoded"] = le_subject.fit_transform(df["Subject"])

    # Preprocessing task 4: encode target column (Learning_Mode)
    le_target = LabelEncoder()  # Online=1, Offline=0
    df["Target"] = le_target.fit_transform(df["Learning_Mode"])
    print(f"Target classes: {le_target.classes_}")  # ['Offline' 'Online']

    # Preprocessing task 5: remove unnecessary original text columns
    df.drop(columns=["Learning_Mode", "Subject"], inplace=True)

    # Preprocessing task 6: split into features and target
    feature_cols = [
        "Study_Hours", "Memory_Retention", "Focus_Level",
        "Exam_Score", "Subject_Encoded", "Study_Efficiency"
    ]
    X = df[feature_cols]
    y = df["Target"]

    # Preprocessing task 7: check final shape
    print(f"Features shape: {X.shape}")
    print(f"Target shape:   {y.shape}")
    print(f"Feature columns: {feature_cols}")

    # NumPy task 3: check class balance using numpy
    unique, counts = np.unique(y.values, return_counts=True)
    print(f"Class balance — {dict(zip(unique, counts))}")

    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing task: scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # NumPy task 4: verify shapes with numpy
    print(f"\nTrain set size: {np.shape(X_train_scaled)}")
    print(f"Test  set size: {np.shape(X_test_scaled)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, le_target


if __name__ == "__main__":
    preprocess("data/dataset.csv")
