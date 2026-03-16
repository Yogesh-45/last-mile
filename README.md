# last-mile

A lightweight Natural Language Understanding (NLU) pipeline for Hinglish (Hindi+English code-mixed) delivery driver messages. It classifies driver intent and extracts structured slot values in under 2 ms per request — no GPU required.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Approach](#approach)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Results](#results)
6. [Qualitative Examples](#qualitative-examples)

---

## Project Structure

```
swg/
├── nlu/                        ← Core ML package (importable)
│   ├── __init__.py             ← Public API exports
│   ├── config.py               ← All constants: paths, hyper-parameters, thresholds
│   ├── preprocessing.py        ← Shared text normalisation
│   └── slot_extractor.py       ← Slot extraction: constant / regex / keyword / ML
├── training/                   ← Offline training & evaluation scripts
│   ├── train.py                ← Train intent classifier + per-slot ML models
│   └── evaluate.py             ← Evaluate on held-out test set, print failures
├── scripts/                    ← Utility scripts
│   ├── benchmark.py            ← Latency & throughput benchmarking
│   └── data_generation.py      ← Synthetic dataset generator
├── models/                     ← All saved ML artefacts (auto-generated)
│   ├── intent/
│   │   ├── intent_classifier.pkl
│   │   └── tfidf_vectorizer.pkl
│   └── slots/
│       ├── clf_order_issue__issue_type.pkl
│       └── ...
├── data/
│   ├── hinglish_delivery_merged.json   ← Training data (~1 100 samples)
│   └── test_instructions.json          ← Held-out test set (50 samples)
├── static/
│   └── index.html              ← Web UI (two-panel, no build step required)
├── app.py                      ← FastAPI entry point (uvicorn app:app)
└── requirements.txt
```

---

## Approach

### Intent Classification

```
Raw text  ──▶  preprocess()  ──▶  TF-IDF FeatureUnion  ──▶  Logistic Regression  ──▶  intent label
                                  (word + char n-grams)      (softmax probabilities)
```

**Text preprocessing** (`preprocessing.py`)
- Lowercase, strip non-alphanumeric characters (keeps digits for "10 min"), collapse whitespace.
- Identical function used at training and inference — no train/serve skew.

**TF-IDF FeatureUnion** (word n-grams + character n-grams)

| Sub-vectorizer | Parameters | Why |
|---|---|---|
| Word n-grams | (1,2), max 4 000, `sublinear_tf` | Captures phrases like "ready nahi", "phone band" |
| Char n-grams | (3,4), max 3 000, `char_wb`, `sublinear_tf` | Makes model robust to Hinglish typos — "custmr" and "customer" share trigrams |

`sublinear_tf=True` applies `log(1+tf)` to dampen stop-word dominance.

**Logistic Regression** with `class_weight="balanced"` handles imbalanced intent distribution.  
**5-fold stratified cross-validation** is used to report a reliable F1 score in addition to a single validation split.

**Confidence threshold** (default `0.40`): predictions below this return `intent="unknown"` to avoid overconfident guesses on out-of-domain input.

**Rule-based intent overrides** fire before the threshold — e.g. the phrase "dial karo" always maps to `call_customer` regardless of model confidence.

---

### Slot Extraction

Slots are extracted using a tiered hybrid approach (priority: top → bottom):

| Priority | Method | Intents |
|---|---|---|
| 1 | **Constant** — hardcoded value, intent alone determines the slot | `call_customer`, `mark_delivered`, `mark_picked_up` |
| 2 | **Regex** — numeric value always present in text | `report_delay` → `delay_time` |
| 3 | **Keyword** — rule-based pattern match (single canonical class) | `navigation_help` → `navigation_action: show_route` |
| 4 | **ML** — TF-IDF + LR classifier trained per (intent, slot) | `order_issue`, `customer_unavailable`, `get_address` |
| 4b | **Keyword fallback** — used if no ML model file exists yet | same intents as above |

ML slot classifiers use the same FeatureUnion architecture as the intent classifier (smaller `max_features` to avoid overfitting on small per-class sample counts).

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models

```bash
python -m training.train
```

This produces `models/intent/intent_classifier.pkl`, `models/intent/tfidf_vectorizer.pkl`, and all files under `models/slots/`.

---

## Usage

### Evaluate on the test set

```bash
python -m training.evaluate
```

Prints a per-intent classification report, slot match accuracy, and per-sample pass/fail. Failures are saved to `evaluate_failures.json`.

### Run the API server

```bash
uvicorn app:app --reload
```

Open **http://localhost:8000** in a browser to use the web UI.

**API endpoints:**

```
POST /predict
  Body: {"text": "customer phone nahi utha raha"}
  Returns: {intent, confidence, slots, top_intents, low_confidence}

GET  /health
  Returns: {status, intents, confidence_threshold, slot_models}
```

**Example curl:**

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "10 minute late ho jayega"}'
```

### Run the benchmark

```bash
python -m scripts.benchmark             # 500 timed runs per intent
python -m scripts.benchmark --runs 200  # faster (less precise)
```

---

## Results

### Test set (50 held-out samples)

| Metric | Score |
|---|---|
| Intent accuracy | **100 %** (50 / 50) |
| Slot match accuracy | **100 %** (50 / 50) |
| 5-fold CV F1 (weighted) | **≈ 0.97** |

### Model sizes

| Artefact | Size |
|---|---|
| `intent_classifier.pkl` | ~148 KB |
| `tfidf_vectorizer.pkl` | ~84 KB |
| `slot_models/` (3 models) | ~156 KB |
| **Total** | **~388 KB** |

### Inference latency (single-threaded, CPU-only)

| Percentile | Latency |
|---|---|
| Median (p50) | ~0.8 ms |
| p95 | ~1.5 ms |
| p99 | ~2.3 ms |
| Throughput (est.) | ~1 000 req/s |

Component breakdown for a typical request:

| Step | Time |
|---|---|
| Text preprocessing | < 0.05 ms |
| TF-IDF vectorization | ~0.6 ms |
| Logistic Regression classify | ~0.1 ms |
| Slot extraction (ML lookup) | ~0.05 ms |

---

## Qualitative Examples

### 1. Customer not picking up the phone

**Input:** `"Customer phone nahi utha raha, call kar liya 3 baar"`

| Field | Value |
|---|---|
| **Intent** | `customer_unavailable` |
| **Confidence** | 0.91 |
| **Slots** | `availability: no_response` |
| **Extraction method** | ML (TF-IDF + LR on customer_unavailable/availability) |

The phrase "phone nahi utha" triggers the `no_response` class. Character n-grams handle common misspellings like "custmr phone" gracefully.

---

### 2. Reporting a delay with a specific time

**Input:** `"Thoda late ho jayega, 10 min aur lag jayenge"`

| Field | Value |
|---|---|
| **Intent** | `report_delay` |
| **Confidence** | 0.97 |
| **Slots** | `delay_time: 10, unit: minutes` |
| **Extraction method** | Regex (`\b(\d+)\s*(?:min\|minute)s?\b`) |

The numeric value "10" is always reliably captured by regex without needing an ML model.

---

### 3. Requesting the next delivery address

**Input:** `"Next order ka address bhejo bhai, kahan jaana hai"`

| Field | Value |
|---|---|
| **Intent** | `get_address` |
| **Confidence** | 0.88 |
| **Slots** | `order_reference: next` |
| **Extraction method** | ML (TF-IDF + LR on get_address/order_reference) |

The keyword "next" clearly signals the upcoming order. The model also handles phrasing like "agle order" correctly via character n-gram overlap.

---

### 4. Reporting a damaged package

**Input:** `"Restaurant ka packet damage ho gaya hai, leak kar raha hai"`

| Field | Value |
|---|---|
| **Intent** | `order_issue` |
| **Confidence** | 0.93 |
| **Slots** | `issue_type: damaged_package` |
| **Extraction method** | ML (TF-IDF + LR on order_issue/issue_type) |

"damage" and "leak" are strong features for `damaged_package`. Balanced class weights in training prevent the majority class from dominating.

---

### 5. Out-of-domain / unknown query

**Input:** `"Aaj mausam bahut accha hai"`  *(Today the weather is very nice)*

| Field | Value |
|---|---|
| **Intent** | `unknown` |
| **Confidence** | 0.22 |
| **Slots** | *(empty)* |
| **low_confidence** | `true` |

The model's top probability (0.22) falls below the confidence threshold (0.25), so it correctly abstains rather than forcing a wrong prediction. This prevents hallucinated slot values for unrecognised inputs.
