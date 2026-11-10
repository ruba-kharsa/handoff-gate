"""Extra evaluation only: compare keywords, TF-IDF, and MiniLM.

All three methods use the same labeled CSV; none changes the main fix.
"""
import argparse
import csv
import re
import time
from pathlib import Path

from handoff_gate_selected import EXAMPLES, MODEL_DIR, MiniLMGate, decide_from_scores

ROOT = Path(__file__).resolve().parent
TESTS_CSV = ROOT / "tests" / "test_cases.csv"

# Simple lexical baseline: words commonly associated with each specialist.
KEYWORDS = {
    "accommodation": [
        "hotel", "hotels", "room", "rooms",
        "accommodation", "lodging",
        "apartment", "apartments", "bed", "beds",
        "breakfast", "check-in", "check-out",
        "hostel", "hostels", "resort", "resorts",
        "guesthouse"
    ],
    "flights": [
        "flight", "flights", "airline", "airlines",
        "plane", "airplane", "aircraft",
        "ticket", "tickets", "airfare",
        "baggage", "luggage", "airport",
        "departure", "departures",
        "arrival", "arrivals",
        "boarding", "layover"
    ],
    "activities": [
        "museum", "museums", "tour", "tours",
        "sightseeing", "attraction", "attractions",
        "excursion", "excursions",
        "guided tour", "guided tours",
        "tourist attractions",
        "things to do", "activities",
        "experience", "experiences"
    ],
    "trips": [
        "itinerary", "itineraries", "vacation",
        "schedule", "trip planning", "travel plan",
        "destinations", "destination",
        "multi-day", "holiday", "holidays",
        "travel itinerary", "trip itinerary",
        "route planning", "travel schedule"
    ],
}


class KeywordGate:
    def decide(self, message, active):
        # Count whole-word matches; no pretrained model or training is required.
        # Forward mixed-domain messages; reject only when active is the sole match.
        scores = {
            domain: sum(bool(re.search(r"\b" + re.escape(word) + r"\b", message,
                                       flags=re.I)) for word in words)
            for domain, words in KEYWORDS.items()
        }
        matches = {domain for domain, score in scores.items() if score > 0}
        return ("REJECT" if matches == {active} else "ALLOW"), scores


class TfidfGate:
    def __init__(self, min_score=0.15):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.min_score = min_score
        self.rows = [(d, phrase) for d, phrases in EXAMPLES.items() for phrase in phrases]
        # TF-IDF compares words and pairs of words, not semantic embeddings.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.references = self.vectorizer.fit_transform([phrase for _, phrase in self.rows])

    def decide(self, message, active):
        from sklearn.metrics.pairwise import cosine_similarity
        # Compare the message with each reference phrase.
        similarities = cosine_similarity(
            self.vectorizer.transform([message]), self.references
        )[0]
        scores = {
            domain: max(float(similarities[i]) for i, (label, _) in enumerate(self.rows)
                        if label == domain)
            for domain in EXAMPLES
        }
        return decide_from_scores(scores, active, self.min_score), scores


def evaluate(method, gate, cases):
    passed = false_reject = false_allow = 0
    elapsed = 0.0
    lines = [f"=== {method.upper()} ==="]
    # Timing excludes one-time initialization, as in the main script.
    for case in cases:
        start = time.perf_counter()
        actual, scores = gate.decide(case["message"], case["specialist"])
        elapsed += time.perf_counter() - start
        expected = case["expected"].strip().upper()
        correct = actual == expected
        passed += correct
        false_reject += actual == "REJECT" and expected == "ALLOW"
        false_allow += actual == "ALLOW" and expected == "REJECT"
        lines.append(f"{'PASS' if correct else 'FAIL'} | {actual:6} expected={expected:6} "
                     f"| {case['specialist']:13} | {case['message']}")
        if not correct:
                      lines.append(f"Scores: { {k: round(v, 3) for k, v in scores.items()} }")
    count = len(cases)
    lines += [f"Results: {passed}/{count} passed | false REJECT={false_reject} "
              f"| false ALLOW={false_allow}",
              f"Average decision time: {elapsed * 1000 / count:.2f} ms" if count
              else "Average decision time: N/A", ""]
    return lines


def main():
    parser = argparse.ArgumentParser(description="Optional comparison of three offline gates")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--method", choices=["keywords", "tfidf", "minilm", "all"], default="all")
    parser.add_argument("--tests-csv", type=Path, default=TESTS_CSV)
    parser.add_argument("--min-score", type=float, default=0, help="MiniLM threshold")
    parser.add_argument("--tfidf-min-score", type=float, default=0.15)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "results" /"results_comparison.txt")
    args = parser.parse_args()
    if not args.test:
        parser.print_help()
        return

    with args.tests_csv.open(newline="", encoding="utf-8-sig") as file:
        cases = list(csv.DictReader(file))

    # Model loading happens only for the method currently being evaluated.
    methods = ["keywords", "tfidf", "minilm"] if args.method == "all" else [args.method]
    lines = [f"Test cases: {len(cases)}", ""]
    for method in methods:
        try:
            start = time.perf_counter()
            if method == "keywords":
                gate = KeywordGate()
            elif method == "tfidf":
                gate = TfidfGate(args.tfidf_min_score)
            else:
                gate = MiniLMGate(args.model_dir, args.min_score)
            init_ms = (time.perf_counter() - start) * 1000
            lines += [f"Initialization time ({method}): {init_ms:.2f} ms"]
            lines += evaluate(method, gate, cases)
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            lines += [f"SKIPPED {method}: {exc}", ""]
    result = "\n".join(lines) + "\n"
    print(result)
    # Store the printed test outcomes and average times for design-note review.
    args.output.write_text(result, encoding="utf-8")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
