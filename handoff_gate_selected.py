"""Main submission: validate specialist handoffs using a local MiniLM model.
https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx

REJECT means the current specialist should handle the request.
ALLOW means the existing router should decide whether to transfer it.
"""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
from light_embed import TextEmbedding

# Keep paths relative to this script so it works from any working directory.
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "minilm_onnx"  # Or supply --model-dir with your downloaded model path.
TESTS_CSV = ROOT / "tests" / "test_cases.csv"

# Short reference phrases describe the work owned by each specialist. Can be enhanced with testing
EXAMPLES = {
    "accommodation": [
        "hotels and room prices",
        "accommodation or lodging for stay",
        "hotel rooms breakfast and parking",
        "affordable apartment bed or beds",
    ],
    "flights": [
        "flights and airline tickets",
        "a flight with different airlines and plane options",
        "flight baggage allowance and luggage restrictions",
        "airport flights departures and arrivals",
    ],
    "activities": [
        "museums, museum visits and tourist attractions",
        "guided tour or sightseeing tours",
        "attraction, excursion or excursions",
    ],
    "trips": [
        "multi-day travel itinerary and itineraries",
        "vacation schedule and destinations",
        "trip planning and travel plan for holiday",
    ],
}

def decide_from_scores(scores, active_specialist, min_score):
    """Only block a handoff for a clear match with the active specialist."""
    # No margin is used: only the winning domain and minimum score matter.
    best = max(scores, key=scores.get)
    if best == active_specialist and scores[best] >= min_score:
        return "REJECT"
    return "ALLOW"


class MiniLMGate:
    def __init__(self, model_dir, min_score=0):
        
        self.min_score = min_score
        # Remember which specialist owns each reference phrase.
        self.rows = [(domain, phrase) for domain, phrases in EXAMPLES.items()
                     for phrase in phrases]
        model_dir = Path(model_dir)
        if not (model_dir / "model.onnx").is_file():
            raise FileNotFoundError(f"Missing model.onnx in: {model_dir}")

        # Load the local model and encode the domain phrases once. calculate the norms directly
        self.model = TextEmbedding(
            model_name_or_path=str(model_dir),
            model_config={"onnx_file": "model.onnx", "normalize": True},
        )
        self.references = np.asarray(
            self.model.encode([phrase for _, phrase in self.rows])
        )
        

    def decide(self, message, active_specialist):
        if active_specialist not in EXAMPLES:
            raise ValueError(f"Unknown specialist: {active_specialist}")

        # Encode only the latest user message; the handoff reason is intentionally ignored.
        vector = np.asarray(self.model.encode([message]))[0]
        # Cosine similarity between the normalized message and each reference phrase.
        # The Dot between the normalized examples embeddings and the normalized message embeddings 
        similarities = self.references @ vector
        # Keep the strongest matching reference phrase for each specialist.
        scores = {
            domain: max(float(similarities[i]) for i, (label, _) in enumerate(self.rows)
                        if label == domain)
            for domain in EXAMPLES
        }
        return decide_from_scores(scores, active_specialist, self.min_score), scores


def run_tests(gate, tests_csv):
    """Run all CSV examples and return printable results and timings."""
    lines = []
    passed = false_reject = false_allow = total = 0
    elapsed = 0.0
    # The CSV is the same test set used by the optional comparison.
    with Path(tests_csv).open(newline="", encoding="utf-8-sig") as file:
        for case in csv.DictReader(file):
            # Time only the decision, not one-time model initialization.
            start = time.perf_counter()
            actual, scores = gate.decide(case["message"], case["specialist"])
            elapsed += time.perf_counter() - start
            expected = case["expected"].strip().upper()
            success = actual == expected
            passed += success
            total += 1
            # False REJECT blocks a valid handoff; false ALLOW sends an
            # unnecessary handoff to the router.
            false_reject += actual == "REJECT" and expected == "ALLOW"
            false_allow += actual == "ALLOW" and expected == "REJECT"
            lines.append(f"{'PASS' if success else 'FAIL'} | {actual:6} expected={expected:6} "
                         f"| {case['specialist']:13} | {case['message']}")
            if not success:
              lines.append(f"Scores: { {k: round(v, 3) for k, v in scores.items()} }")
    lines += [f"\nResults: {passed}/{total} passed | false REJECT={false_reject} "
              f"| false ALLOW={false_allow}",
              f"Average decision time: {elapsed * 1000 / total:.2f} ms" if total
              else "Average decision time: N/A"]
    return lines


def main():
    parser = argparse.ArgumentParser(description="Selected offline MiniLM handoff gate")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tests-csv", type=Path, default=TESTS_CSV)
    parser.add_argument("--message")
    parser.add_argument("--specialist", choices=list(EXAMPLES))
    parser.add_argument("--min-score", type=float, default=0)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "results_selected.txt",
                        help="Text file where results are saved")
    args = parser.parse_args()
    if (args.message is None) != (args.specialist is None):
        parser.error("Provide --message and --specialist together")
    if not args.test and args.message is None:
        parser.print_help()
        return

    # Measure model loading separately from request processing.
    start = time.perf_counter()
    gate = MiniLMGate(args.model_dir, args.min_score)
    init_ms = (time.perf_counter() - start) * 1000
    lines = ["Selected method: MiniLM", f"Minimum score: {args.min_score}",
             f"Initialization time: {init_ms:.2f} ms", ""]
    if args.test:
        lines.extend(run_tests(gate, args.tests_csv))
    if args.message is not None:
        start = time.perf_counter()
        decision, scores = gate.decide(args.message, args.specialist)
        request_ms = (time.perf_counter() - start) * 1000
        lines += ["", f"Message: {args.message}", f"Specialist: {args.specialist}",
                  f"Decision: {decision}", f"Time: {request_ms:.2f} ms",
                  f"Scores: { {k: round(v, 3) for k, v in scores.items()} }"]
    result = "\n".join(lines) + "\n"
    print(result)
    # Save the same human-readable output that was printed to the terminal.
    args.output.write_text(result, encoding="utf-8")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
