# Handoff Gate — Main Submission and Extra Comparison

## Problem and goal

A specialist sometimes calls the **handoff** tool even when the user's latest message is about that specialist's own subject. For example, the accommodation specialist might request a handoff when the user asks for a different hotel room. This causes an unnecessary router call and may incorrectly change the conversation's active specialist.

The existing orchestrator already has a router that can select the same specialist again. **This fix does not change the router.** Instead, a small gate checks a handoff request before it reaches the router:

- **REJECT:** The latest message best matches the currently active specialist and its similarity score meets the minimum threshold. The server should let that specialist continue without another handoff attempt in the same turn.
- **ALLOW:** Otherwise, forward the handoff request to the existing router. `ALLOW` is not a decision to transfer to a new specialist.

The prototype uses only the **latest user message** and **active specialist**, not the specialist-generated handoff reason or previous messages. Unclear cases go to the router, assuming it receives enough conversation context to decide reliably. This is an English-only, offline demonstration rather than a production integration. The **default minimum similarity score is 0**.

## Installation and Setup

### 1. Clone the Repository

Clone the project from GitHub and navigate to its directory:

```bash
git clone https://github.com/ruba-kharsa/handoff-gate.git
cd handoff-gate
```
### 2. Create a Virtual Environment (recommended)

It is recommended to create a Python virtual environment before installing the project dependencies. This keeps the required packages isolated from your system Python installation and avoids conflicts with other projects.

Open a terminal in the project directory and run:

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

**Windows (Git Bash):**
```bash
source .venv/Scripts/activate
```

**Windows (Command Prompt):**
```bash
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

Once activated, `(.venv)` should appear in your terminal prompt.

### 3. Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Deactivate When Finished running the scripts

```bash
deactivate
```

## Main submission (selected fix)

**`handoff_gate_selected.py` is the main submission.** It uses a small pretrained **MiniLM ONNX** sentence-embedding model through LightEmbed, with NumPy for cosine similarity. Model weights must already be available locally. There is **no live LLM, API key, network request, or model download in the script**. 
https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx

The model is only 91 MB and is ok for this quick fix.
### Files required for the main submission

| File | Purpose |
| --- | --- |
| `handoff_gate_selected.py` | Selected gate, test runner, CLI, and timing |
| `test_cases.csv` | Shared expected decisions (106 examples) |
| `requirements.txt` | Main Python dependencies |
| `minilm_onnx/` (supplied locally) | ONNX model and tokenizer/config files; included in the ZIP |

**Run the main submission:**

```bash
python handoff_gate_selected.py --test 
python handoff_gate_selected.py --test --min-score 0.2
python handoff_gate_selected.py --message "Find cheaper rooms" --specialist accommodation 
```

The script prints case-level PASS/FAIL results, false REJECT/ALLOW counts, **initialization time**, and **average decision time**. It saves the same output to `results/results_selected.txt` (override with `--output path/to/file.txt`). These files are created when you run the tests; measured results are not prefilled.

## Extra submission (comparison)

**`handoff_comparison.py` is optional supporting work, not the selected handoff fix.** It compares three *independent* approaches against the same CSV:

| Method | Technology | Extra requirements |
| --- | --- | --- |
| Keywords | Python standard library (`re`) | None |
| TF-IDF | scikit-learn | `scikit-learn` (also installs its dependencies) |
| MiniLM | Same selected gate as the main submission | `light-embed`, `numpy`, and local model files |

### Files required for the extra comparison

| File | Purpose |
| --- | --- |
| `handoff_comparison.py` | Run keywords, TF-IDF, and MiniLM independently |
| `handoff_gate_selected.py` | Imported for shared domain examples and MiniLM gate |
| `test_cases.csv` | Identical expected decisions for all three methods |
| `requirements_comparison.txt` | Installs main requirements and scikit-learn |
| `minilm_onnx/` (supplied locally) | Needed **only** when comparing MiniLM |

To run all three approaches:

```bash
python -m pip install -r requirements_comparison.txt
python handoff_comparison.py --test 
```

Or test only one approach:

```bash
python handoff_comparison.py --test --method keywords
python handoff_comparison.py --test --method tfidf
python handoff_comparison.py --test --method minilm 
```

Outputs are printed and saved to `results/results_comparison.txt` by default. Both scripts support `--tests-csv` and `--output`. The **comparison results and rationale for choosing MiniLM are discussed in the Design Notes**; the extra script is included only so reviewers can reproduce the comparison.

A local **Ollama LLM validator was explored but excluded** from the submitted comparison: it was too resource-intensive to run the full challenge suite on the available 8 GB computer, and the task specifically asks for no live model.

## Test data and interpretation

- `main_test_cases.csv` — 20 difficult examples
- `test_cases.csv` — 106 examples (default input)

You can use any of these CSVs with `--tests-csv`, for example:

```bash
python handoff_gate_selected.py --test --tests-csv tests/main_test_cases.csv
```

Similarity scores are **not probabilities**; the `0` threshold is experimental and can be increased. The script times decisions on the test cases (not an isolated benchmark).
