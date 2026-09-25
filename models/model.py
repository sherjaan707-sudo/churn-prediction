"""Reusable pipeline code for the customer churn project.

The notebook imports from here so experiments and the final model share
exactly the same data cleaning and preprocessing steps.

Run directly to train the final model and print test-set results:
    python models/model.py
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "telco_customer_churn.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_data(path=DATA_PATH):
    """Load the raw CSV and apply the minimal cleaning every experiment needs."""
    df = pd.read_csv(path)
    # TotalCharges is stored as text; 11 new customers (tenure 0) have a blank value.
    # They have not been billed yet, so 0 is the correct value rather than a guess.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def split_data(df, test_size=0.2):
    """Stratified train/test split so both sets keep the ~26.5% churn rate."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=RANDOM_STATE)


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def get_candidate_models():
    """All models compared in the experiment notebook."""
    return {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1),
    }


def build_pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def build_final_model():
    """The model chosen in decision_log.txt (Decision 1)."""
    return build_pipeline(LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))


def evaluate(pipeline, X, y):
    """Return the metrics used throughout the project as a dict."""
    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    final = build_final_model().fit(X_train, y_train)
    for name, value in evaluate(final, X_test, y_test).items():
        print(f"{name:>10}: {value:.3f}")
