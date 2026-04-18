"""
main.py
-------
Entry point for the Support Ticket Classification System.

Orchestrates the full ML pipeline in three phases:
  1. Training  — preprocess data, extract features, train & save models
  2. Evaluation — report accuracy, plot confusion matrices & comparisons
  3. Demo       — run the prediction pipeline on sample tickets
"""

import os
import sys

# ── Ensure imports resolve from the project root ────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        Support Ticket Classification System  v1.0           ║
║           Category · Priority · Routing                      ║
╚══════════════════════════════════════════════════════════════╝
"""


def section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


# ===========================================================================
# STEP 1 — Training
# ===========================================================================
def run_training():
    section("STEP 1 — Training")
    from src.train import main as train_main
    X_test, y_cat_test, y_pri_test = train_main()
    return X_test, y_cat_test, y_pri_test


# ===========================================================================
# STEP 2 — Evaluation
# ===========================================================================
def run_evaluation(X_test, y_cat_test, y_pri_test):
    section("STEP 2 — Evaluation")
    import joblib
    from src.evaluate import evaluate_classifier, compare_models

    models_dir = os.path.join(PROJECT_ROOT, "models")

    category_model   = joblib.load(os.path.join(models_dir, "category_model.pkl"))
    priority_model   = joblib.load(os.path.join(models_dir, "priority_model.pkl"))
    category_encoder = joblib.load(os.path.join(models_dir, "category_encoder.pkl"))
    priority_encoder = joblib.load(os.path.join(models_dir, "priority_encoder.pkl"))

    cat_results = evaluate_classifier(
        category_model, X_test, y_cat_test, category_encoder, "Category Classifier"
    )
    pri_results = evaluate_classifier(
        priority_model, X_test, y_pri_test, priority_encoder, "Priority Classifier"
    )

    compare_models({
        "Category Classifier": cat_results["accuracy"],
        "Priority Classifier": pri_results["accuracy"],
    })

    return cat_results["accuracy"], pri_results["accuracy"]


# ===========================================================================
# STEP 3 — Prediction demo
# ===========================================================================
def run_demo():
    section("STEP 3 — Prediction Demo")
    from src.predict import TicketClassifier

    classifier = TicketClassifier()

    demo_tickets = [
        "I was charged twice on my credit card this month. Please refund the extra charge.",
        "The mobile app crashes every time I try to open my profile settings.",
        "I need to reset my password because I no longer have access to my old email.",
    ]

    for i, ticket in enumerate(demo_tickets, 1):
        result = classifier.predict(ticket)
        print(f"\n  [Ticket {i}]")
        print(f"  Text     : {result['original_text'][:75]}…")
        print(f"  Category : {result['category']}  (conf: {result['category_confidence']:.2%})")
        print(f"  Priority : {result['priority']}  (conf: {result['priority_confidence']:.2%})")
        print(f"  Action   : {result['recommendation']}")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print(BANNER)

    # ── Phase 1: Train ───────────────────────────────────────────────────────
    X_test, y_cat_test, y_pri_test = run_training()

    # ── Phase 2: Evaluate ────────────────────────────────────────────────────
    cat_acc, pri_acc = run_evaluation(X_test, y_cat_test, y_pri_test)

    # ── Phase 3: Demo ────────────────────────────────────────────────────────
    run_demo()

    # ── Summary ──────────────────────────────────────────────────────────────
    section("FINAL SUMMARY")
    print(f"  ✅ Category Classifier Accuracy : {cat_acc:.4f}  ({cat_acc:.2%})")
    print(f"  ✅ Priority  Classifier Accuracy : {pri_acc:.4f}  ({pri_acc:.2%})")
    print(f"\n  Model artifacts saved → models/")
    print(f"  Charts saved          → models/*.png")
    print(f"\n  Run `python src/predict.py` independently for more demos.")
    print(f"\n{'═' * 64}\n")
