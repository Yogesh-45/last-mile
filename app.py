"""
FastAPI server — serves the trained NLU pipeline over HTTP.

Endpoints
---------
  GET  /          → Web UI (static/index.html)
  POST /predict   → {text} → {intent, confidence, slots, top_intents, low_confidence}
  GET  /health    → loaded model metadata

Usage
-----
  uvicorn app:app --reload          # development
  uvicorn app:app --host 0.0.0.0    # production
"""

from typing import Any

import joblib  # pyright: ignore[reportMissingImports]
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nlu.config import (
    CONFIDENCE_THRESHOLD,
    INTENT_CLF_PATH,
    STATIC_DIR,
    TOP_N_INTENTS,
    VECTORIZER_PATH,
)
from nlu.preprocessing import preprocess
from nlu.slot_extractor import correct_intent, extract_slots, load_slot_models

# ── Load models once at startup ──────────────────────────────────────
if not INTENT_CLF_PATH.exists() or not VECTORIZER_PATH.exists():
    raise RuntimeError(
        "Model artefacts not found. Run `python -m training.train` first."
    )

intent_model  = joblib.load(INTENT_CLF_PATH)
vectorizer    = joblib.load(VECTORIZER_PATH)
slot_models   = load_slot_models()
INTENT_LABELS = list(intent_model.classes_)

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="last-mile", version="1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str


class IntentScore(BaseModel):
    intent: str
    confidence: float


class PredictResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    slots: dict[str, Any]
    top_intents: list[IntentScore]
    low_confidence: bool


# ── Endpoints ────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Run the full NLU pipeline on a single driver message.

    Pipeline
    --------
    1. Preprocess text
    2. TF-IDF vectorize
    3. Logistic Regression → intent probabilities
    4. Rule-based intent correction
    5. Confidence threshold → 'unknown' for out-of-domain input
    6. Slot extraction (constant / regex / keyword / ML)
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    cleaned    = preprocess(text)
    vec        = vectorizer.transform([cleaned])
    proba      = intent_model.predict_proba(vec)[0]
    top_idx    = proba.argsort()[::-1]

    best_intent     = INTENT_LABELS[top_idx[0]]
    best_confidence = float(proba[top_idx[0]])

    # Rule-based override fires before threshold
    corrected = correct_intent(text, best_intent)

    low_confidence = best_confidence < CONFIDENCE_THRESHOLD
    if low_confidence:
        final_intent = "unknown"
        slots: dict  = {}
    else:
        final_intent = corrected
        slots        = extract_slots(text, final_intent, slot_models)

    top_intents = [
        IntentScore(intent=INTENT_LABELS[i], confidence=round(float(proba[i]), 4))
        for i in top_idx[:TOP_N_INTENTS]
    ]

    return PredictResponse(
        text=text,
        intent=final_intent,
        confidence=round(best_confidence, 4),
        slots=slots,
        top_intents=top_intents,
        low_confidence=low_confidence,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "intents": INTENT_LABELS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "slot_models": {k: list(v.keys()) for k, v in slot_models.items()},
    }
