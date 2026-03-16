"""
Train the intent classifier and per-slot ML classifiers.

Produces
--------
  models/intent/intent_classifier.pkl  – Logistic Regression intent model
  models/intent/tfidf_vectorizer.pkl   – FeatureUnion TF-IDF vectorizer
  models/slots/                        – Per-intent, per-slot model pairs

Usage (from project root)
--------------------------
  python -m training.train
"""

import json
import shutil
from collections import defaultdict

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import FeatureUnion, Pipeline

from nlu.config import (
    CHAR_MAX_FEATURES,
    CV_FOLDS,
    INTENT_CLF_PATH,
    INTENT_MAX_ITER,
    INTENT_MODELS_DIR,
    RANDOM_STATE,
    SKIP_SLOT_INTENTS,
    SLOT_CHAR_MAX_FEATURES,
    SLOT_MAX_ITER,
    SLOT_MODELS_DIR,
    SLOT_WORD_MAX_FEATURES,
    TEST_SIZE,
    TRAIN_DATA_PATH,
    VECTORIZER_PATH,
    WORD_MAX_FEATURES,
)
from nlu.preprocessing import preprocess
from nlu.slot_extractor import CONSTANT_SLOTS, slot_model_path


def build_vectorizer() -> FeatureUnion:
    """
    Combined TF-IDF vectorizer: word n-grams + character n-grams.

    Word n-grams  capture phrases like 'ready nahi', 'phone band'.
    Char n-grams  tolerate Hinglish typos — 'custmr' and 'customer'
                  share character trigrams 'cus', 'ust', etc.
    sublinear_tf  applies log(1+tf) to dampen high-frequency filler words.
    char_wb       pads each word so n-grams don't bleed across boundaries.
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=WORD_MAX_FEATURES,
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=1,
            max_features=CHAR_MAX_FEATURES,
            sublinear_tf=True,
        )),
    ])


def train_intent_classifier(df: pd.DataFrame) -> tuple:
    """Train and return (vectorizer, model, X_val, y_val, y_pred_val)."""
    X_train, X_val, y_train, y_val = train_test_split(
        df["text_clean"],
        df["intent"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["intent"],
    )
    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    vectorizer  = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec   = vectorizer.transform(X_val)

    n_word = vectorizer.transformer_list[0][1].get_feature_names_out().shape[0]
    n_char = vectorizer.transformer_list[1][1].get_feature_names_out().shape[0]
    print(f"Features: {n_word} word + {n_char} char = {n_word + n_char} total")

    model = LogisticRegression(max_iter=INTENT_MAX_ITER, class_weight="balanced")
    model.fit(X_train_vec, y_train)
    return vectorizer, model, X_val, y_val, model.predict(X_val_vec)


def cross_validate(df: pd.DataFrame, vectorizer: FeatureUnion) -> None:
    """Run stratified k-fold CV and print weighted F1 scores."""
    pipeline = Pipeline([
        ("vec", vectorizer),
        ("clf", LogisticRegression(max_iter=INTENT_MAX_ITER, class_weight="balanced")),
    ])
    cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, df["text_clean"], df["intent"], cv=cv, scoring="f1_weighted",
    )
    print(f"{CV_FOLDS}-Fold CV → F1 weighted: {scores.mean():.4f} ± {scores.std():.4f}  "
          f"(per fold: {[round(s, 3) for s in scores]})")


def train_slot_classifiers(data: list[dict]) -> None:
    """
    Train one (vectorizer, classifier) pair per (intent, slot) combination.

    Skips intents that use constant / regex / keyword extraction.
    Also skips slots with only one unique label.
    """
    skip = SKIP_SLOT_INTENTS | set(CONSTANT_SLOTS.keys())

    intent_to_samples: dict = defaultdict(list)
    for row in data:
        intent_to_samples[row["intent"]].append(row)

    print("\nTraining slot classifiers...")
    for intent, samples in sorted(intent_to_samples.items()):
        if intent in skip:
            print(f"  Skipping {intent}  (constant / keyword / regex extractor)")
            continue

        slot_keys: set = {k for s in samples for k in s.get("slots", {}).keys()}

        for slot_key in sorted(slot_keys):
            texts_s, labels_s = [], []
            for s in samples:
                val = s.get("slots", {}).get(slot_key)
                if val is not None:
                    texts_s.append(preprocess(s["text"]))
                    labels_s.append(str(val))

            if len(set(labels_s)) < 2:
                only = labels_s[0] if labels_s else "none"
                print(f"  Skipping {intent}/{slot_key}: only one class ({only!r})")
                continue

            slot_vec = FeatureUnion([
                ("word", TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=1,
                    max_features=SLOT_WORD_MAX_FEATURES, sublinear_tf=True,
                )),
                ("char", TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 4), min_df=1,
                    max_features=SLOT_CHAR_MAX_FEATURES, sublinear_tf=True,
                )),
            ])
            X_slot   = slot_vec.fit_transform(texts_s)
            slot_clf = LogisticRegression(max_iter=SLOT_MAX_ITER, class_weight="balanced")
            slot_clf.fit(X_slot, labels_s)

            vec_path, clf_path = slot_model_path(intent, slot_key)
            joblib.dump(slot_vec, vec_path)
            joblib.dump(slot_clf, clf_path)
            print(f"  Saved {intent}/{slot_key}  classes={slot_clf.classes_.tolist()}")

    print("Slot classifier training complete.")


def main() -> None:
    with open(TRAIN_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {TRAIN_DATA_PATH.name}")

    df = pd.DataFrame(data)
    df["text_clean"] = df["text"].apply(preprocess)

    # Intent model
    vectorizer, model, X_val, y_val, y_pred_val = train_intent_classifier(df)

    # Save intent artefacts
    INTENT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,      INTENT_CLF_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved → {INTENT_CLF_PATH.relative_to(INTENT_CLF_PATH.parents[2])}, "
          f"{VECTORIZER_PATH.relative_to(VECTORIZER_PATH.parents[2])}")

    # Internal validation
    print("\nValidation classification report:")
    print(classification_report(y_val, y_pred_val))

    # Cross-validation
    cross_validate(df, vectorizer)

    # Slot models — wipe stale files first
    if SLOT_MODELS_DIR.exists():
        shutil.rmtree(SLOT_MODELS_DIR)
    SLOT_MODELS_DIR.mkdir(parents=True)
    train_slot_classifiers(data)


if __name__ == "__main__":
    main()
