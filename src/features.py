"""
src/features.py
---------------
Feature engineering module for the Support Ticket Classifier.

Responsibilities:
  - Convert preprocessed text into sparse numerical feature matrices
    using either TF-IDF or Bag-of-Words (CountVectorizer)
  - Encode string class labels to integers and decode them back
  - Persist / reload fitted vectorisers to/from disk
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder


class FeatureExtractor:
    """
    Wraps a sklearn text vectoriser (TF-IDF or BoW) with save/load support.

    Args:
        method:       'tfidf' or 'bow'  (default: 'tfidf')
        max_features: Vocabulary size cap (default: 5000)
    """

    def __init__(self, method: str = "tfidf", max_features: int = 5000) -> None:
        self.method = method
        self.max_features = max_features

        # Initialise the appropriate vectoriser
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # unigrams + bigrams
                min_df=2,            # ignore very rare terms
            )
        elif method == "bow":
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
            )
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose 'tfidf' or 'bow'.")

    # ------------------------------------------------------------------
    def fit_transform(self, texts: list):
        """
        Fit the vectoriser on *texts* and return the feature matrix.

        Args:
            texts: List of preprocessed ticket strings.

        Returns:
            Sparse feature matrix  (n_samples × max_features).
        """
        return self.vectorizer.fit_transform(texts)

    # ------------------------------------------------------------------
    def transform(self, texts: list):
        """
        Transform *texts* using an already-fitted vectoriser.

        ⚠️  Call fit_transform() first — this method does NOT refit.

        Args:
            texts: List of preprocessed ticket strings.

        Returns:
            Sparse feature matrix.
        """
        return self.vectorizer.transform(texts)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Persist the fitted vectoriser to disk.

        Args:
            path: File path (e.g. 'models/tfidf_vectorizer.pkl').
        """
        joblib.dump(self.vectorizer, path)
        print(f"[FeatureExtractor] Vectoriser saved → {path}")

    # ------------------------------------------------------------------
    def load(self, path: str) -> None:
        """
        Load a previously saved vectoriser from disk.

        Args:
            path: Path to the .pkl file.
        """
        self.vectorizer = joblib.load(path)
        print(f"[FeatureExtractor] Vectoriser loaded ← {path}")


# ------------------------------------------------------------------
# Label encoding helpers
# ------------------------------------------------------------------

def encode_labels(series: pd.Series):
    """
    Encode a string-label Series to integer array.

    Args:
        series: pd.Series of class name strings.

    Returns:
        Tuple (encoded_array, fitted_LabelEncoder).
    """
    le = LabelEncoder()
    encoded = le.fit_transform(series)
    return encoded, le


def decode_labels(encoded, label_encoder: LabelEncoder) -> list:
    """
    Convert integer labels back to their original string names.

    Args:
        encoded:       Integer array / list of encoded labels.
        label_encoder: A fitted LabelEncoder instance.

    Returns:
        List of original class name strings.
    """
    return list(label_encoder.inverse_transform(encoded))


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "cannot login account",
        "billing charge error",
        "reset my password",
    ]
    fe = FeatureExtractor(method="tfidf")
    X = fe.fit_transform(sample_texts)
    print("Feature matrix shape:", X.shape)
