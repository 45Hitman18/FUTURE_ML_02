"""
notebooks/02_model_analysis.py
------------------------------
Post-training model analysis script.

Loads saved model artifacts and produces:
  1. Top-20 TF-IDF features per category class (Logistic Regression coefficients)
  2. Per-class precision/recall heatmap saved to models/class_pr_heatmap.png
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# ===========================================================================
# Load artifacts
# ===========================================================================
print("[analysis] Loading saved model artifacts …")
try:
    category_model   = joblib.load(os.path.join(MODELS_DIR, "category_model.pkl"))
    tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    category_encoder = joblib.load(os.path.join(MODELS_DIR, "category_encoder.pkl"))
    priority_model   = joblib.load(os.path.join(MODELS_DIR, "priority_model.pkl"))
    priority_encoder = joblib.load(os.path.join(MODELS_DIR, "priority_encoder.pkl"))
except FileNotFoundError as e:
    print(f"❌ Could not load artifacts: {e}")
    print("   Run 'python main.py' first to train and save the models.")
    sys.exit(1)

# ===========================================================================
# 1. Top-20 TF-IDF features per category
# ===========================================================================
print("\n── Top-20 TF-IDF features per category ─────────────────────────────────")
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
class_names   = category_encoder.classes_
coef_matrix   = category_model.coef_   # shape: (n_classes, n_features)

TOP_N = 20

for i, cls in enumerate(class_names):
    top_indices = np.argsort(coef_matrix[i])[-TOP_N:][::-1]
    top_features = feature_names[top_indices]
    print(f"\n  [{cls}]")
    print("  " + ", ".join(top_features))

# ===========================================================================
# 2. Per-class precision / recall heatmap
# ===========================================================================
print("\n[analysis] Building class-wise PR heatmap …")

# Reload test data from the training split (re-runs quickly)
from src.train import load_and_prepare_data, DATA_PATH, TEXT_COLUMN, CATEGORY_COLUMN
from src.preprocess import TextPreprocessor
from src.features import encode_labels
from sklearn.model_selection import train_test_split

df, priority_col = load_and_prepare_data(DATA_PATH)
preprocessor = TextPreprocessor()
df["processed_text"] = preprocessor.preprocess_column(df, TEXT_COLUMN)

y_category, _ = encode_labels(df[CATEGORY_COLUMN].astype(str))
_, X_text_test, _, y_cat_test = train_test_split(
    df["processed_text"].tolist(),
    y_category,
    test_size=0.20,
    random_state=42,
    stratify=y_category,
)
X_test = tfidf_vectorizer.transform(X_text_test)

y_pred = category_model.predict(X_test)
y_pred_labels = category_encoder.inverse_transform(y_pred)
y_test_labels = category_encoder.inverse_transform(y_cat_test)

# Build per-class precision/recall DataFrame
report_dict = classification_report(
    y_test_labels, y_pred_labels,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)
pr_df = pd.DataFrame(report_dict).T.loc[class_names, ["precision", "recall", "f1-score"]]
pr_df = pr_df.astype(float)

# Plot heatmap
fig, ax = plt.subplots(figsize=(8, max(5, len(class_names) * 0.6 + 2)))
sns.heatmap(
    pr_df,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    ax=ax,
    linewidths=0.5,
)
ax.set_title("Per-Class Precision / Recall / F1 — Category Classifier", fontsize=13, pad=12)
ax.set_ylabel("Category", fontsize=11)
ax.set_xlabel("Metric", fontsize=11)
plt.tight_layout()

save_path = os.path.join(MODELS_DIR, "class_pr_heatmap.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"[analysis] PR heatmap saved → {save_path}")
print("\n[analysis] Done ✓")
