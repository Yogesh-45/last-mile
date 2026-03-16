"""
nlu — core NLU package for the Delivery Assistant.

Public API
----------
  preprocess(text)                           → cleaned str
  extract_slots(text, intent, slot_models)   → {slot: value}
  load_slot_models()                         → {intent: {slot: (vec, clf)}}
  correct_intent(text, predicted)            → intent str
"""

from nlu.preprocessing import preprocess
from nlu.slot_extractor import (
    CONSTANT_SLOTS,
    correct_intent,
    extract_slots,
    load_slot_models,
    slot_model_path,
)

__all__ = [
    "preprocess",
    "CONSTANT_SLOTS",
    "correct_intent",
    "extract_slots",
    "load_slot_models",
    "slot_model_path",
]
