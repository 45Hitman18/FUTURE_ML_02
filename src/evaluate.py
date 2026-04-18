"""
src/evaluate.py
---------------
Evaluation and visualisation module for the Support Ticket Classifier.

Provides:
  - evaluate_classifier()  — accuracy, classification report, confusion matrix
  - compare_models()       — horizontal bar chart of multiple model accuracies
  - show_misclassified()   — tabular view of prediction errors
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")


# ===========================================================================
def evaluate_classifier(
    model,
    X_test,
    y_test,
    label_encoder,
    model_name: str,
) -> dict:
    """
    Evaluate a classifier and save a confusion-matrix heatmap.

    Args:
        model:         Fitted sklearn estimator.
        X_test:        Sparse test feature matrix.
        y_test:        Integer-encoded ground-truth labels.
        label_encoder: Fitted LabelEncoder (used for decoding + class names).
        model_name:    Human-readable name used in titles and filenames.

    Returns:
        dict with keys 'accuracy' (float) and 'report' (str).
    """
    # ── Predict ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test_labels,
        y_pred_labels,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    # ── Console output ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Overall Accuracy : {acc:.4f}")
    print(f"\n{report}")

    # ── Confusion matrix heatmap ─────────────────────────────────────────────
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(max(8, len(label_encoder.classes_)), max(6, len(label_encoder.classes_) - 2)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax,
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()

    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_name = model_name.replace(" ", "_").replace("/", "-")
    save_path = os.path.join(MODELS_DIR, f"{safe_name}_confusion_matrix.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [evaluate] Confusion matrix saved → {save_path}")

    return {"accuracy": acc, "report": report}


# ===========================================================================
def compare_models(results_dict: dict) -> None:
    """
    Plot a horizontal bar chart comparing model accuracies and print a table.

    Args:
        results_dict: {model_name (str): accuracy (float)}
    """
    names      = list(results_dict.keys())
    accuracies = [results_dict[n] for n in names]

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 1.2)))
    bars = ax.barh(names, accuracies, color="#4C72B0", edgecolor="white", height=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=14)
    ax.axvline(0.5, color="red", linewidth=0.8, linestyle="--", label="50% baseline")
    ax.legend(fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(MODELS_DIR, "model_comparison.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n[evaluate] Model comparison chart saved → {save_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print(f"{'Model':<30} {'Accuracy':>8}")
    print("─" * 40)
    for name, acc in zip(names, accuracies):
        print(f"{name:<30} {acc:>8.4f}")
    print("─" * 40)


# ===========================================================================
def show_misclassified(
    model,
    X_test,
    y_test,
    texts_test: list,
    label_encoder,
    n: int = 10,
) -> None:
    """
    Print the first *n* misclassified examples as a readable table.

    Args:
        model:         Fitted sklearn estimator.
        X_test:        Sparse test feature matrix.
        y_test:        Integer-encoded ground-truth labels.
        texts_test:    Original (preprocessed) text strings aligned with X_test.
        label_encoder: Fitted LabelEncoder for decoding.
        n:             Number of misclassified examples to display.
    """
    y_pred = model.predict(X_test)

    # Identify misclassified indices
    wrong_mask = y_pred != y_test
    wrong_idx  = np.where(wrong_mask)[0][:n]

    if len(wrong_idx) == 0:
        print("  🎉 No misclassified examples found!")
        return

    print(f"\n[evaluate] Showing {len(wrong_idx)} misclassified examples\n")
    col1, col2, col3 = 55, 20, 20
    header = f"{'Text':<{col1}} {'Actual':<{col2}} {'Predicted':<{col3}}"
    print(header)
    print("─" * (col1 + col2 + col3))

    for idx in wrong_idx:
        text      = str(texts_test[idx])[:col1 - 2]
        actual    = label_encoder.inverse_transform([y_test[idx]])[0]
        predicted = label_encoder.inverse_transform([y_pred[idx]])[0]
        print(f"{text:<{col1}} {actual:<{col2}} {predicted:<{col3}}")


# ===========================================================================
if __name__ == "__main__":
    print("Run this module via main.py after training.")
