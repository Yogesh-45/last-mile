"""
Central configuration for the Delivery NLU pipeline.

All file paths, hyper-parameters, and behavioural constants live here.
Every other module imports from this file — nothing is hard-coded elsewhere.
"""

from pathlib import Path

# Project root (two levels up: nlu/config.py → nlu/ → project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Directory layout ─────────────────────────────────────────────────
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"          # top-level models folder
STATIC_DIR  = BASE_DIR / "static"

# Sub-directories inside models/
INTENT_MODELS_DIR = MODELS_DIR / "intent"  # intent classifier + vectorizer
SLOT_MODELS_DIR   = MODELS_DIR / "slots"   # per-slot classifiers

# ── Data files ───────────────────────────────────────────────────────
TRAIN_DATA_PATH = DATA_DIR / "hinglish_delivery_merged.json"
TEST_DATA_PATH  = DATA_DIR / "test_instructions.json"

# ── Saved model artefacts ────────────────────────────────────────────
INTENT_CLF_PATH = INTENT_MODELS_DIR / "intent_classifier.pkl"
VECTORIZER_PATH = INTENT_MODELS_DIR / "tfidf_vectorizer.pkl"
FAILURES_PATH   = BASE_DIR / "evaluate_failures.json"

# ── Training hyper-parameters ────────────────────────────────────────
TEST_SIZE    = 0.2   # fraction held out for internal validation
RANDOM_STATE = 42
CV_FOLDS     = 5     # stratified k-fold for reliable accuracy estimate

# Intent vectorizer (FeatureUnion: word n-grams + character n-grams)
WORD_MAX_FEATURES = 4000
CHAR_MAX_FEATURES = 3000

# Per-slot vectorizer (smaller to avoid overfitting on few samples)
SLOT_WORD_MAX_FEATURES = 2000
SLOT_CHAR_MAX_FEATURES = 1500

# Logistic Regression
INTENT_MAX_ITER = 1000
SLOT_MAX_ITER   = 500

# ── Slot extraction ──────────────────────────────────────────────────
# Intents handled by constant / regex / keyword logic — no ML model trained
SKIP_SLOT_INTENTS: set[str] = {
    "report_delay",    # delay_time extracted by regex
    "navigation_help", # navigation_action is always show_route
}

# ── Serving ──────────────────────────────────────────────────────────
# Predictions below this threshold are returned as intent="unknown"
CONFIDENCE_THRESHOLD: float = 0.25
TOP_N_INTENTS: int = 5
