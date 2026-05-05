"""
model.py
--------
Trains a Decision Tree Classifier to predict Learning_Mode (Online vs Offline).

Why Decision Tree?
  - Easy to interpret (we can understand which features matter most)
  - Works well for classification with mixed numerical and categorical features
  - No strong distributional assumptions
"""

from sklearn.tree import DecisionTreeClassifier


def train_model(X_train, y_train, random_state: int = 42):
    """Train a Decision Tree Classifier and return the fitted model."""

    model = DecisionTreeClassifier(
        max_depth=5,          # limit depth to avoid overfitting
        random_state=random_state,
        class_weight="balanced",  # handle any slight class imbalance
    )
    model.fit(X_train, y_train)
    print("Model trained: DecisionTreeClassifier (max_depth=5)")
    return model


if __name__ == "__main__":
    # Quick smoke-test
    from preprocessing import preprocess
    X_train, X_test, y_train, y_test, features, le = preprocess("data/dataset.csv")
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Sample predictions (first 10): {preds[:10]}")
