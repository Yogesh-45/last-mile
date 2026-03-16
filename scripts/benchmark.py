"""
Benchmark the full NLU inference pipeline: latency, throughput, and
per-component breakdown.

Usage (from project root)
--------------------------
  python -m scripts.benchmark              # 500 timed runs per input
  python -m scripts.benchmark --runs 1000  # higher run count
"""

import argparse
import statistics
import time

import joblib

from nlu.config import INTENT_CLF_PATH, VECTORIZER_PATH
from nlu.preprocessing import preprocess
from nlu.slot_extractor import correct_intent, extract_slots, load_slot_models

# One representative sentence per intent
TEST_INPUTS: list[tuple[str, str]] = [
    ("call_customer",        "Customer ka number dial karo"),
    ("customer_unavailable", "Customer phone nahi utha raha"),
    ("get_address",          "next order ka address bhejo bhai"),
    ("mark_delivered",       "Order deliver ho gaya mark kar do"),
    ("mark_picked_up",       "food pick kar liya mark pickup"),
    ("navigation_help",      "Route dikhao bhai customer location ka"),
    ("order_issue",          "Restaurant bol raha order ready nahi hai"),
    ("report_delay",         "Thoda late ho jayega 10 min lag jayenge"),
]


# ── Helpers ──────────────────────────────────────────────────────────

def percentile(sorted_data: list[float], p: float) -> float:
    idx = min(int(len(sorted_data) * p / 100), len(sorted_data) - 1)
    return sorted_data[idx]


def make_pipeline(model, vectorizer, slot_models):
    def run(text: str) -> float:
        t0      = time.perf_counter()
        cleaned = preprocess(text)
        vec     = vectorizer.transform([cleaned])
        proba   = model.predict_proba(vec)[0]
        intent  = correct_intent(text, model.classes_[proba.argmax()])
        _       = extract_slots(text, intent, slot_models)
        return (time.perf_counter() - t0) * 1000
    return run


def time_runs(fn, text: str, n_warmup: int, n_runs: int) -> list[float]:
    for _ in range(n_warmup):
        fn(text)
    return [fn(text) for _ in range(n_runs)]


def print_summary(times: list[float]) -> None:
    s = sorted(times)
    rows = [
        ("Average (mean)",    f"{statistics.mean(times):.3f} ms"),
        ("Median (p50)",      f"{statistics.median(times):.3f} ms"),
        ("Best  (min)",       f"{min(times):.3f} ms"),
        ("Worst (max)",       f"{max(times):.3f} ms"),
        ("p75",               f"{percentile(s, 75):.3f} ms"),
        ("p90",               f"{percentile(s, 90):.3f} ms"),
        ("p95",               f"{percentile(s, 95):.3f} ms"),
        ("p99",               f"{percentile(s, 99):.3f} ms"),
        ("Std deviation",     f"{statistics.stdev(times):.3f} ms"),
        ("Throughput (est.)", f"{1000/statistics.mean(times):.0f} req/s  (single-threaded)"),
    ]
    for label, value in rows:
        print(f"  {label:<26}  {value}")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NLU pipeline benchmark")
    parser.add_argument("--runs",   type=int, default=500)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    N_RUNS, N_WARMUP = args.runs, args.warmup

    print("Loading models...", end=" ", flush=True)
    t0          = time.perf_counter()
    model       = joblib.load(INTENT_CLF_PATH)
    vectorizer  = joblib.load(VECTORIZER_PATH)
    slot_models = load_slot_models()
    print(f"done in {(time.perf_counter() - t0)*1000:.0f} ms\n")

    run_pipeline = make_pipeline(model, vectorizer, slot_models)

    # ── Per-intent table ─────────────────────────────────────────
    SEP    = "=" * 66
    header = f"{'avg':>8}  {'median':>8}  {'p95':>8}  {'p99':>8}  {'max':>8}"
    print(f"{SEP}\nPER-INTENT LATENCY  ({N_RUNS} runs, {N_WARMUP} warmup)\n{SEP}")
    print(f"\n  {'Intent':<26}  {header}\n  {'-'*72}")

    all_times: list[float] = []
    for intent_label, text in TEST_INPUTS:
        times = time_runs(run_pipeline, text, N_WARMUP, N_RUNS)
        all_times.extend(times)
        s = sorted(times)
        print(f"  {intent_label:<26}  {statistics.mean(times):>7.2f}ms  "
              f"{statistics.median(times):>7.2f}ms  {percentile(s,95):>7.2f}ms  "
              f"{percentile(s,99):>7.2f}ms  {max(times):>7.2f}ms")

    # ── Overall summary ──────────────────────────────────────────
    print(f"\n{SEP}\nOVERALL SUMMARY  ({len(all_times)} total runs)\n{SEP}\n")
    print_summary(all_times)

    # ── Component breakdown ──────────────────────────────────────
    sample_text = TEST_INPUTS[2][1]
    print(f"\n{SEP}\nCOMPONENT BREAKDOWN  ({N_RUNS} runs, '{sample_text}')\n{SEP}\n")

    t_pre, t_vec, t_clf, t_slot = [], [], [], []
    for _ in range(N_WARMUP):
        run_pipeline(sample_text)

    for _ in range(N_RUNS):
        t0 = time.perf_counter(); cleaned = preprocess(sample_text)
        t_pre.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter(); vec = vectorizer.transform([cleaned])
        t_vec.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        proba  = model.predict_proba(vec)[0]
        intent = correct_intent(sample_text, model.classes_[proba.argmax()])
        t_clf.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter(); _ = extract_slots(sample_text, intent, slot_models)
        t_slot.append((time.perf_counter() - t0) * 1000)

    def med(lst): return statistics.median(lst)

    components = [
        ("1. Preprocess",       med(t_pre)),
        ("2. TF-IDF vectorize", med(t_vec)),
        ("3. Intent LR",        med(t_clf)),
        ("4. Slot extraction",  med(t_slot)),
    ]
    total = sum(ms for _, ms in components)

    def bar(frac: float, width: int = 32) -> str:
        filled = round(frac * width)
        return "█" * filled + "░" * (width - filled)

    for name, ms in components:
        frac = ms / total
        print(f"  {name:<22}  {ms:6.3f} ms  {frac*100:4.1f}%  {bar(frac)}")
    print(f"  {'─'*72}\n  {'TOTAL':<22}  {total:6.3f} ms")


if __name__ == "__main__":
    main()
