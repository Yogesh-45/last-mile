"""
Text preprocessing shared across training, evaluation, and serving.

All pipeline components use this single function so that text is
transformed identically at training time and inference time.
"""

import re


def preprocess(text: str) -> str:
    """
    Normalise raw Hinglish driver text before vectorisation.

    Steps
    -----
    1. Lowercase              — removes case sensitivity.
    2. Strip non-alphanumeric — removes punctuation and symbols.
       Digits are kept because delay minutes ("10 min") are meaningful.
    3. Collapse whitespace    — single space between tokens.

    Examples
    --------
    >>> preprocess("Customer phone nahi utha raha!")
    'customer phone nahi utha raha'
    >>> preprocess("custmr  ko  CALL  kar do.")
    'custmr ko call kar do'
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
