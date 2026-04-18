"""
src/train.py
------------
Model training script for the Support Ticket Classifier.

Trains TWO classifiers:
  - Model A: Ticket Category Classifier  (e.g. Billing, Technical, Account)
  - Model B: Priority Level Classifier   (High / Medium / Low)

Both models share the same TF-IDF feature matrix derived from the
preprocessed ticket description column.

Saved artifacts (models/ folder):
  - category_model.pkl
  - priority_model.pkl
  - tfidf_vectorizer.pkl
  - category_encoder.pkl
  - priority_encoder.pkl
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Resolve imports whether run as a script or as a module
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import TextPreprocessor
from src.features import FeatureExtractor, encode_labels

# ---------------------------------------------------------------------------
# ★  CONFIGURATION — update these constants if your CSV column names differ
# ---------------------------------------------------------------------------
TEXT_COLUMN     = "Ticket Description"  # column containing ticket text
CATEGORY_COLUMN = "Ticket Type"         # column containing category label
PRIORITY_COLUMN = "Ticket Priority"     # column containing priority  (or None)
DATA_PATH       = os.path.join(PROJECT_ROOT, "data", "customer_support_tickets.csv")
MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")


# ===========================================================================
# STEP 1 — Data loading & preparation
# ===========================================================================
def load_and_prepare_data(filepath: str):
    """
    Load the CSV, handle nulls, and ensure a priority column exists.

    If PRIORITY_COLUMN is not found in the CSV, a synthetic priority column
    is created with weights: High=20 %, Medium=50 %, Low=30 %.

    Args:
        filepath: Path to the CSV dataset.

    Returns:
        Tuple (df, effective_priority_column_name).
    """
    print(f"[train] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[train] Raw shape: {df.shape}")

    # Drop rows with missing ticket text
    df = df.dropna(subset=[TEXT_COLUMN])
    print(f"[train] After dropping null text rows: {df.shape}")

    # ── Priority column ──────────────────────────────────────────────────────
    if PRIORITY_COLUMN and PRIORITY_COLUMN in df.columns:
        print(f"[train] Found priority column: '{PRIORITY_COLUMN}'")
        df[PRIORITY_COLUMN] = df[PRIORITY_COLUMN].fillna("Medium")
        effective_priority_col = PRIORITY_COLUMN
    else:
        print("[train] Priority column not found — generating synthetic priorities.")
        np.random.seed(42)
        df["priority"] = np.random.choice(
            ["High", "Medium", "Low"],
            size=len(df),
            p=[0.20, 0.50, 0.30],
        )
        effective_priority_col = "priority"

    return df, effective_priority_col


# ===========================================================================
# STEP 2 — Model selection helper
# ===========================================================================
def train_model(X_train, y_train, model_type: str = "logreg"):
    """
    Train and return a fitted classifier.

    Args:
        X_train:    Sparse feature matrix (training set).
        y_train:    Integer-encoded label array.
        model_type: One of 'logreg', 'nb', 'rf'.

    Returns:
        A fitted sklearn estimator.
    """
    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "nb":
        model = MultinomialNB()
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose logreg / nb / rf.")

    print(f"[train] Training {model_type} …")
    model.fit(X_train, y_train)
    return model


# ===========================================================================
# STEP 3 — Main pipeline
# ===========================================================================
def main():
    """
    End-to-end training pipeline.

    Returns:
        Tuple (X_test, y_test_category, y_test_priority) for downstream
        evaluation in main.py / evaluate.py.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 3a. Load & prepare data ──────────────────────────────────────────────
    df, priority_col = load_and_prepare_data(DATA_PATH)

    # ── 3b. Preprocess ticket text ───────────────────────────────────────────
    print("[train] Preprocessing ticket text …")
    preprocessor = TextPreprocessor()
    df["processed_text"] = preprocessor.preprocess_column(df, TEXT_COLUMN)

    # ── 3c. Encode labels ────────────────────────────────────────────────────
    y_category, category_encoder = encode_labels(df[CATEGORY_COLUMN].astype(str))
    y_priority,  priority_encoder  = encode_labels(df[priority_col].astype(str))

    # ── 3d. Train / test split (stratify on category) ───────────────────────
    print("[train] Splitting into train/test (80/20) …")
    (
        X_text_train, X_text_test,
        y_cat_train,  y_cat_test,
        y_pri_train,  y_pri_test,
    ) = train_test_split(
        df["processed_text"].tolist(),
        y_category,
        y_priority,
        test_size=0.20,
        random_state=42,
        stratify=y_category,
    )

    # ── 3e. TF-IDF feature extraction ───────────────────────────────────────
    print("[train] Extracting TF-IDF features …")
    fe = FeatureExtractor(method="tfidf", max_features=5000)
    X_train = fe.fit_transform(X_text_train)
    X_test  = fe.transform(X_text_test)

    # ── 3f. Train classifiers ────────────────────────────────────────────────
    category_model = train_model(X_train, y_cat_train, model_type="logreg")
    priority_model  = train_model(X_train, y_pri_train, model_type="logreg")

    # ── 3g. Save all artifacts ───────────────────────────────────────────────
    print("[train] Saving model artifacts …")
    joblib.dump(category_model,   os.path.join(MODELS_DIR, "category_model.pkl"))
    joblib.dump(priority_model,   os.path.join(MODELS_DIR, "priority_model.pkl"))
    joblib.dump(fe.vectorizer,    os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(category_encoder, os.path.join(MODELS_DIR, "category_encoder.pkl"))
    joblib.dump(priority_encoder, os.path.join(MODELS_DIR, "priority_encoder.pkl"))
    print(f"[train] All artifacts saved to: {MODELS_DIR}/")

    # ── 3h. Report training accuracy ─────────────────────────────────────────
    cat_train_acc = category_model.score(X_train, y_cat_train)
    pri_train_acc  = priority_model.score(X_train,  y_pri_train)
    print(f"\n[train] Category Classifier — training accuracy : {cat_train_acc:.4f}")
    print(f"[train] Priority  Classifier — training accuracy : {pri_train_acc:.4f}")

    return X_test, y_cat_test, y_pri_test


# ===========================================================================
if __name__ == "__main__":
    main()
