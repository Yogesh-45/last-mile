"""
Evaluate the trained pipeline on the held-out test set.

Requires
--------
  models/intent/intent_classifier.pkl  (produced by training/train.py)
  models/intent/tfidf_vectorizer.pkl   (produced by training/train.py)
  models/slots/                        (produced by training/train.py)

Usage (from project root)
--------------------------
  python -m training.evaluate
"""

import json

import joblib
from sklearn.metrics import classification_report, confusion_matrix

from nlu.config import (
    FAILURES_PATH,
    INTENT_CLF_PATH,
    TEST_DATA_PATH,
    VECTORIZER_PATH,
)
from nlu.preprocessing import preprocess
from nlu.slot_extractor import correct_intent, extract_slots, load_slot_models


# ── Helpers ──────────────────────────────────────────────────────────

def slots_match(true_slots: dict, pred_slots: dict) -> bool:
    """Return True if every key in true_slots matches the prediction."""
    return all(str(pred_slots.get(k, "")) == str(v) for k, v in true_slots.items())


def predict_batch(
    texts: list[str],
    model,
    vectorizer,
    slot_models: dict,
) -> tuple[list[str], list[dict]]:
    """Run the full pipeline on a list of raw texts."""
    cleaned = [preprocess(t) for t in texts]
    X       = vectorizer.transform(cleaned)

    raw_intents  = model.predict(X)
    pred_intents = [correct_intent(t, i) for t, i in zip(texts, raw_intents)]
    pred_slots   = [
        extract_slots(t, intent, slot_models)
        for t, intent in zip(texts, pred_intents)
    ]
    return pred_intents, pred_slots


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    model       = joblib.load(INTENT_CLF_PATH)
    vectorizer  = joblib.load(VECTORIZER_PATH)
    slot_models = load_slot_models()
    print(f"Slot classifiers loaded: {list(slot_models.keys()) or 'none'}")

    with open(TEST_DATA_PATH, encoding="utf-8") as f:
        test_data = json.load(f)

    texts          = [r["text"]          for r in test_data]
    y_true_intents = [r["intent"]        for r in test_data]
    y_true_slots   = [r.get("slots", {}) for r in test_data]

    y_pred_intents, y_pred_slots = predict_batch(
        texts, model, vectorizer, slot_models
    )

    SEP = "=" * 62

    # ── Intent metrics ────────────────────────────────────────────
    print(f"\n{SEP}\nINTENT CLASSIFICATION REPORT\n{SEP}")
    print(classification_report(y_true_intents, y_pred_intents))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true_intents, y_pred_intents, labels=model.classes_))
    print("Labels:", list(model.classes_))

    # ── Slot accuracy ─────────────────────────────────────────────
    n_correct = sum(slots_match(t, p) for t, p in zip(y_true_slots, y_pred_slots))
    n_total   = len(test_data)
    print(f"\nSlot match accuracy: {n_correct}/{n_total} = {n_correct/n_total:.2%}")

    # ── Per-sample results ────────────────────────────────────────
    failures_intent: list[dict] = []
    failures_slots:  list[dict] = []

    print(f"\n{SEP}\nPER-SAMPLE RESULTS\n{SEP}")
    for i, (text, true_intent, pred_intent, true_slots, pred_slots) in enumerate(
        zip(texts, y_true_intents, y_pred_intents, y_true_slots, y_pred_slots)
    ):
        intent_ok = true_intent == pred_intent
        slot_ok   = slots_match(true_slots, pred_slots)
        status    = (
            "OK" if (intent_ok and slot_ok)
            else ("INTENT_FAIL" if not intent_ok else "SLOT_FAIL")
        )
        print(f"[{i:2d}] {status}")
        print(f"      text   : {text!r}")
        print(f"      intent : true={true_intent}, pred={pred_intent}  {'✓' if intent_ok else '✗'}")
        print(f"      slots  : true={true_slots}")
        print(f"               pred={pred_slots}  {'✓' if slot_ok else '✗'}")

        if not intent_ok:
            failures_intent.append({
                "index": i, "text": text,
                "true_intent": true_intent, "predicted_intent": pred_intent,
                "true_slots": true_slots, "predicted_slots": pred_slots,
            })
        if not slot_ok:
            failures_slots.append({
                "index": i, "text": text,
                "intent": pred_intent,
                "true_slots": true_slots, "predicted_slots": pred_slots,
            })

    # ── Failure summaries ─────────────────────────────────────────
    print(f"\n{SEP}\nINTENT FAILURES: {len(failures_intent)}/{n_total}\n{SEP}")
    if failures_intent:
        for f in failures_intent:
            print(f"  [{f['index']}] {f['text']!r}")
            print(f"       true: {f['true_intent']}  →  pred: {f['predicted_intent']}")
    else:
        print("  None — all intents correct.")

    print(f"\n{SEP}\nSLOT FAILURES: {len(failures_slots)}/{n_total}\n{SEP}")
    if failures_slots:
        for f in failures_slots:
            print(f"  [{f['index']}] {f['text']!r}")
            print(f"       intent   : {f['intent']}")
            print(f"       true     : {f['true_slots']}")
            print(f"       predicted: {f['predicted_slots']}")
    else:
        print("  None — all slots correct.")

    # ── Save failures ─────────────────────────────────────────────
    with open(FAILURES_PATH, "w", encoding="utf-8") as fout:
        json.dump(
            {"intent_failures": failures_intent, "slot_failures": failures_slots},
            fout, indent=2, ensure_ascii=False,
        )
    print(f"\nFailures saved → {FAILURES_PATH.name}")


if __name__ == "__main__":
    main()
