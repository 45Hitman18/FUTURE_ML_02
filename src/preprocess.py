"""
src/preprocess.py
-----------------
Text preprocessing module for the Support Ticket Classifier.

Responsibilities:
  - Clean raw ticket text (remove URLs, emails, punctuation, numbers)
  - Tokenize and remove stopwords
  - Lemmatize tokens using WordNetLemmatizer
  - Expose a single preprocess() entry-point that returns a clean string
  - Expose preprocess_column() for DataFrame-level batch processing
"""

import re
import string
import nltk
import pandas as pd

# Download required NLTK resources (safe to call multiple times)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """
    A reusable NLP preprocessing pipeline for customer support ticket text.

    Usage:
        preprocessor = TextPreprocessor()
        clean = preprocessor.preprocess("I can't login to my account!")
    """

    def __init__(self) -> None:
        """Initialise stopword set and lemmatizer."""
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    # ------------------------------------------------------------------
    # Step 1 — raw text cleaning
    # ------------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """
        Remove noise from raw text.

        Steps (in order):
          1. Lowercase
          2. Strip URLs  (http / https)
          3. Strip email addresses
          4. Strip punctuation
          5. Strip digits
          6. Collapse multiple whitespace characters

        Args:
            text: A raw ticket string.

        Returns:
            A cleaned, lowercase string with noise removed.
        """
        if not isinstance(text, str):
            return ""

        # 1. Lowercase
        text = text.lower()

        # 2. Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # 3. Remove email addresses
        text = re.sub(r"\S+@\S+", " ", text)

        # 4. Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # 5. Remove numbers
        text = re.sub(r"\d+", " ", text)

        # 6. Collapse extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ------------------------------------------------------------------
    # Step 2 — tokenisation + stopword removal
    # ------------------------------------------------------------------
    def tokenize(self, text: str) -> list:
        """
        Tokenize text and remove stopwords / very short tokens.

        Args:
            text: A cleaned string (output of clean_text).

        Returns:
            List of meaningful word tokens.
        """
        tokens = word_tokenize(text)

        # Keep tokens that are not stopwords and are at least 2 chars long
        tokens = [
            t for t in tokens
            if t not in self.stop_words and len(t) >= 2
        ]
        return tokens

    # ------------------------------------------------------------------
    # Step 3 — lemmatisation
    # ------------------------------------------------------------------
    def lemmatize(self, tokens: list) -> list:
        """
        Reduce each token to its base (lemma) form.

        Args:
            tokens: List of string tokens.

        Returns:
            List of lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def preprocess(self, text: str) -> str:
        """
        Run the full preprocessing pipeline on a single text string.

        Pipeline: clean_text → tokenize → lemmatize → join

        Args:
            text: Raw ticket text.

        Returns:
            A single preprocessed string ready for feature extraction.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        lemmatized = self.lemmatize(tokens)
        return " ".join(lemmatized)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def preprocess_column(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Apply preprocess() to every row of a DataFrame column.

        Args:
            df:     Source DataFrame.
            column: Name of the column containing raw ticket text.

        Returns:
            A pd.Series of preprocessed strings (same index as df).
        """
        return df[column].fillna("").apply(self.preprocess)


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Hello! I can't log into my account. "
        "Please help ASAP!! Visit http://support.com or email us at help@example.com"
    )
    preprocessor = TextPreprocessor()
    print("Original :", sample)
    print("Cleaned  :", preprocessor.preprocess(sample))
