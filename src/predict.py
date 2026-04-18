"""
src/predict.py
--------------
Prediction pipeline for the Support Ticket Classifier.

Loads all pre-trained artifacts from the models/ folder and exposes:
  - TicketClassifier.predict(text)        → single-ticket prediction dict
  - TicketClassifier.predict_batch(texts) → DataFrame of predictions

Each prediction includes category, priority, confidence scores, and a
human-readable routing recommendation.
"""

import os
import sys
import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import TextPreprocessor

# Routing recommendations keyed by priority level
_PRIORITY_ADVICE = {
    "High":   "🔴 High priority — respond within 2 hours.",
    "Medium": "🟡 Medium priority — respond within 24 hours.",
    "Low":    "🟢 Low priority — respond within 72 hours.",
}


class TicketClassifier:
    """
    End-to-end inference pipeline for a single ticket or a batch.

    Loads artifacts from *models_dir* on initialisation.
    """

    def __init__(self, models_dir: str = "models/") -> None:
        """
        Load all five saved artifacts from disk.

        Args:
            models_dir: Path to the directory that contains the .pkl files.
        """
        # Resolve to absolute path (works whether called from project root or src/)
        if not os.path.isabs(models_dir):
            models_dir = os.path.join(PROJECT_ROOT, models_dir)

        try:
            self.category_model   = joblib.load(os.path.join(models_dir, "category_model.pkl"))
            self.priority_model   = joblib.load(os.path.join(models_dir, "priority_model.pkl"))
            self.tfidf_vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
            self.category_encoder = joblib.load(os.path.join(models_dir, "category_encoder.pkl"))
            self.priority_encoder = joblib.load(os.path.join(models_dir, "priority_encoder.pkl"))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model artifact not found: {e}\n"
                "Run 'python main.py' first to train and save the models."
            )

        self.preprocessor = TextPreprocessor()
        print("[predict] TicketClassifier initialised ✓")

    # ------------------------------------------------------------------
    def predict(self, ticket_text: str) -> dict:
        """
        Classify a single raw ticket string.

        Args:
            ticket_text: Raw, unprocessed customer support ticket text.

        Returns:
            Dictionary with keys: original_text, processed_text,
            category, category_confidence, priority,
            priority_confidence, recommendation.
        """
        # 1. Preprocess
        processed = self.preprocessor.preprocess(ticket_text)

        # 2. Vectorise (single sample — reshape to 1-row matrix)
        X = self.tfidf_vectorizer.transform([processed])

        # 3. Category prediction + confidence
        cat_probs  = self.category_model.predict_proba(X)[0]
        cat_idx    = int(cat_probs.argmax())
        category   = self.category_encoder.classes_[cat_idx]
        cat_conf   = round(float(cat_probs[cat_idx]), 4)

        # 4. Priority prediction + confidence
        pri_probs  = self.priority_model.predict_proba(X)[0]
        pri_idx    = int(pri_probs.argmax())
        priority   = self.priority_encoder.classes_[pri_idx]
        pri_conf   = round(float(pri_probs[pri_idx]), 4)

        # 5. Build human-readable recommendation
        advice     = _PRIORITY_ADVICE.get(priority, "Respond promptly.")
        recommendation = f"Route to {category} team. {advice}"

        return {
            "original_text":         ticket_text,
            "processed_text":        processed,
            "category":              category,
            "category_confidence":   cat_conf,
            "priority":              priority,
            "priority_confidence":   pri_conf,
            "recommendation":        recommendation,
        }

    # ------------------------------------------------------------------
    def predict_batch(self, texts: list) -> pd.DataFrame:
        """
        Classify a list of ticket strings.

        Args:
            texts: List of raw ticket text strings.

        Returns:
            pd.DataFrame where each row is one ticket's prediction.
        """
        results = [self.predict(t) for t in texts]
        return pd.DataFrame(results)


# ===========================================================================
# Demo / self-test
# ===========================================================================
if __name__ == "__main__":
    sample_tickets = [
        "I was charged twice for my last invoice. Please refund the extra amount immediately.",
        "My internet connection keeps dropping every 10 minutes. Restarted the router — still broken.",
        "I forgot my password and the reset email never arrives. Checked spam folder.",
        "I'd like to upgrade my subscription to the Pro plan effective next month.",
        "Your service has been down for 3 hours and I'm losing business. This is unacceptable.",
    ]

    classifier = TicketClassifier()

    print("\n" + "=" * 70)
    print("  DEMO — Ticket Classification Results")
    print("=" * 70)

    for i, ticket in enumerate(sample_tickets, 1):
        result = classifier.predict(ticket)
        print(f"\n[Ticket {i}]")
        print(f"  Text      : {result['original_text'][:80]}…")
        print(f"  Category  : {result['category']}  (conf: {result['category_confidence']:.2%})")
        print(f"  Priority  : {result['priority']}  (conf: {result['priority_confidence']:.2%})")
        print(f"  Action    : {result['recommendation']}")
