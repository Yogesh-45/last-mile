"""
Slot extraction for every supported intent.

Extraction strategy (applied in priority order):
  1. Constant  – slot value never changes for that intent → hardcoded dict.
  2. Regex     – numeric value always present in text (e.g. delay minutes).
  3. Keyword   – rule-based pattern matching for simple categorical slots.
  4. ML        – per-slot TF-IDF + Logistic Regression classifier loaded from
                 disk (trained by training/train.py). Keyword functions serve
                 as fallback when no model file exists yet.
"""

import re
import joblib  # pyright: ignore[reportMissingImports]

from nlu.config import SLOT_MODELS_DIR
from nlu.preprocessing import preprocess


# ── 1. Constant slots ────────────────────────────────────────────────
# Intents where the slot value is fully determined by the intent itself.
CONSTANT_SLOTS: dict[str, dict] = {
    "call_customer":  {"target": "customer"},
    "mark_delivered": {"status": "delivered"},
    "mark_picked_up": {"status": "picked_up"},
}


# ── 2. Regex-based slots ─────────────────────────────────────────────
_DELAY_RE = re.compile(r"\b(\d+)\s*(?:min|minute)s?\b", re.IGNORECASE)


def _extract_report_delay_slots(text: str) -> dict:
    """Extract delay_time (integer minutes) from raw text."""
    m = _DELAY_RE.search(text)
    return {
        "delay_time": int(m.group(1)) if m else None,
        "unit": "minutes",
    }


# ── 3. Keyword-based slots ───────────────────────────────────────────

def _extract_navigation_help_slots(text: str) -> dict:  # noqa: ARG001
    """navigation_action is always show_route (dataset standardised)."""
    return {"navigation_action": "show_route"}


def _extract_get_address_slots_fallback(text: str) -> dict:
    """
    Keyword fallback for get_address when no ML model is present.
    order_reference: current | next
    """
    t = text.lower()
    if any(kw in t for kw in ["current", "abhi", "is order", "ye wala", "yahi"]):
        return {"order_reference": "current"}
    return {"order_reference": "next"}


def _extract_customer_unavailable_slots(text: str) -> dict:
    """
    Keyword fallback for customer_unavailable.
    availability: unreachable | no_response | not_found

    Checked most-specific → least-specific to avoid false positives.
    """
    t = text.lower()
    if any(kw in t for kw in ["reachable nahi", "unreachable", "switched off",
                               "phone band", "band hai", "lag nahi raha"]):
        return {"availability": "unreachable"}
    if any(kw in t for kw in ["phone nahi utha", "call pick", "ring nahi",
                               "utha nahi", "receive nahi", "pick nahi kar",
                               "respond nahi"]):
        return {"availability": "no_response"}
    if any(kw in t for kw in ["mil nahi", "ghar pe nahi", "door pe",
                               "nahi mila", "missing", "address pe nahi",
                               "gate band", "koi nahi"]):
        return {"availability": "not_found"}
    return {"availability": "unreachable"}


def _extract_order_issue_slots_fallback(text: str) -> dict:
    """
    Keyword fallback for order_issue when no ML model is present.
    issue_type: not_ready | damaged_package | order_problem
    """
    t = text.lower()
    if any(kw in t for kw in ["ready nahi", "delay kar raha", "aur lagenge"]):
        return {"issue_type": "not_ready"}
    if any(kw in t for kw in ["damage", "leak", "khula", "open lag"]):
        return {"issue_type": "damaged_package"}
    return {"issue_type": "order_problem"}


# ── 4. ML slot model helpers ─────────────────────────────────────────

def slot_model_path(intent: str, slot: str) -> tuple:
    """
    Return (vectorizer_path, classifier_path) for a given intent + slot.
    Uses '__' as separator so multi-word names parse cleanly.
    """
    safe = intent.replace(" ", "_")
    return (
        SLOT_MODELS_DIR / f"vec_{safe}__{slot}.pkl",
        SLOT_MODELS_DIR / f"clf_{safe}__{slot}.pkl",
    )


def load_slot_models() -> dict:
    """
    Load all saved ML slot models from models/slots/.

    Returns
    -------
    dict : {intent: {slot_name: (vectorizer, classifier)}}
    """
    models: dict = {}
    if not SLOT_MODELS_DIR.exists():
        return models
    for clf_file in SLOT_MODELS_DIR.glob("clf_*__*.pkl"):
        stem    = clf_file.stem[4:]       # strip leading "clf_"
        sep_idx = stem.find("__")
        if sep_idx == -1:
            continue
        intent   = stem[:sep_idx]
        slot     = stem[sep_idx + 2:]
        vec_file = SLOT_MODELS_DIR / f"vec_{stem}.pkl"
        if not vec_file.exists():
            continue
        if intent not in models:
            models[intent] = {}
        models[intent][slot] = (joblib.load(vec_file), joblib.load(clf_file))
    return models


# ── Slot value normalisation ─────────────────────────────────────────
# Maps dialect variants / aliases to their canonical value.
_SLOT_VALUE_NORMALISE: dict[str, dict[str, dict[str, str]]] = {}


def _normalise_slot_value(intent: str, slot: str, value: str) -> str:
    return (
        _SLOT_VALUE_NORMALISE
        .get(intent, {})
        .get(slot, {})
        .get(value, value)
    )


# ── Intent post-processing ───────────────────────────────────────────
# Rule-based overrides applied on top of the ML intent prediction.
# Format: ([trigger_keywords], forced_intent)
_INTENT_OVERRIDE_RULES: list[tuple[list[str], str]] = [
    (["dial karo", "dial kar", "number dial"], "call_customer"),
]


def correct_intent(text: str, predicted_intent: str) -> str:
    """
    Apply rule-based keyword overrides to the ML intent prediction.
    Overrides fire before the confidence threshold so explicit phrases
    like 'dial karo' always win regardless of model confidence.
    """
    t = text.lower()
    for keywords, forced_intent in _INTENT_OVERRIDE_RULES:
        if any(kw in t for kw in keywords):
            return forced_intent
    return predicted_intent


# ── Main extraction entry point ──────────────────────────────────────

def extract_slots(text: str, intent: str, slot_models: dict | None = None) -> dict:
    """
    Extract slots from raw text given the predicted intent.

    Priority order
    --------------
    constant → regex → keyword (navigation) → ML → keyword fallback

    Parameters
    ----------
    text        : Raw (unprocessed) input text.
    intent      : Predicted intent label.
    slot_models : Dict loaded by load_slot_models(); pass None to skip ML.
    """
    if intent in CONSTANT_SLOTS:
        return CONSTANT_SLOTS[intent].copy()

    if intent == "report_delay":
        return _extract_report_delay_slots(text)

    if intent == "navigation_help":
        return _extract_navigation_help_slots(text)

    if slot_models and intent in slot_models:
        cleaned = preprocess(text)
        return {
            slot_name: _normalise_slot_value(
                intent, slot_name,
                str(clf.predict(vec.transform([cleaned]))[0])
            )
            for slot_name, (vec, clf) in slot_models[intent].items()
        }

    # Keyword fallbacks (used before train.py has been run)
    if intent == "get_address":
        return _extract_get_address_slots_fallback(text)
    if intent == "customer_unavailable":
        return _extract_customer_unavailable_slots(text)
    if intent == "order_issue":
        return _extract_order_issue_slots_fallback(text)

    return {}
