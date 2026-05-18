import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from answer import (
    fetch_context_baseline, fetch_context, fetch_context_crossencoder,
    fetch_context_hybrid, generate_answer, client, model
)

BI_ENCODER = "nomic-ai/nomic-embed-text-v1.5"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-large"

RESULTS_DIR = Path("results")

PIPELINES = {
    "baseline":      (fetch_context_baseline, 0),
    "llm_reranker":  (fetch_context, 0),
    "cross_encoder": (fetch_context_crossencoder, 0),
    "hybrid":        (fetch_context_hybrid, 0),
}


class TestQuestion(BaseModel):
    question: str
    keywords: list[str]
    reference_answer: str
    category: str


class JudgeScore(BaseModel):
    feedback: str
    accuracy: float
    completeness: float
    relevance: float


def load_tests(path="tests.jsonl"):
    tests = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tests.append(TestQuestion(**json.loads(line)))
    return tests


def calculate_mrr(keyword, chunks):
    """Reciprocal rank for a single keyword across retrieved chunks."""
    for i, chunk in enumerate(chunks):
        if keyword.lower() in chunk.page_content.lower():
            return 1 / (i + 1)
    return 0.0


def calculate_dcg(relevances, k):
    """Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def calculate_ndcg(keyword, chunks, k=10):
    """nDCG for a single keyword (binary relevance)."""
    relevances = [1 if keyword.lower() in chunk.page_content.lower() else 0 for chunk in chunks[:k]]
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def keyword_coverage(chunks, keywords):
    """Fraction of keywords that appear anywhere in the retrieved chunks."""
    combined = " ".join(chunk.page_content.lower() for chunk in chunks)
    hits = sum(1 for kw in keywords if kw.lower() in combined)
    return hits / len(keywords) if keywords else 0.0


def evaluate_answer(question, reference_answer, pipeline_answer):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """You are an expert evaluator assessing RAG answer quality.
Score the pipeline answer vs the reference on three dimensions (1-5):
- accuracy: how factually correct is it (1=wrong, 5=perfectly accurate — any wrong answer must score 1)
- completeness: how thoroughly does it cover all aspects (5 only if ALL reference info is included)
- relevance: how directly does it answer without extra fluff (5 only if completely on-topic with no extra info)

Reply with ONLY this format — feedback on one line, then scores:
FEEDBACK: <one sentence evaluation>
SCORES: accuracy,completeness,relevance
Example:
FEEDBACK: Answer correctly explains MRR but omits the averaging step mentioned in the reference.
SCORES: 4,3,5"""},
            {"role": "user", "content": f"Question: {question}\n\nReference: {reference_answer}\n\nPipeline answer: {pipeline_answer}"}
        ]
    )
    raw = response.choices[0].message.content.strip()

    feedback = ""
    accuracy, completeness, relevance = 0.0, 0.0, 0.0

    for line in raw.splitlines():
        if line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()
        elif line.startswith("SCORES:"):
            scores_str = line.replace("SCORES:", "").strip()
            parts = [x.strip() for x in scores_str.split(',') if x.strip()]
            if len(parts) == 3:
                try:
                    accuracy, completeness, relevance = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    pass

    return JudgeScore(feedback=feedback, accuracy=accuracy, completeness=completeness, relevance=relevance)


def print_model_info():
    print("\n--- Models ---")
    print(f"  Bi-encoder:    {BI_ENCODER}")
    print(f"  Cross-encoder: {CROSS_ENCODER_MODEL}")
    print(f"  Inference LLM: {model}")
    print()


def debug_pipeline(name):
    fetch_fn, _ = PIPELINES[name]
    tests = load_tests()
    print_model_info()
    print(f"=== DEBUG: {name} ===\n")

    for i, test in enumerate(tests):
        print(f"[{i+1}/{len(tests)}] {test.question}")
        chunks = fetch_fn(test.question)
        mrr_scores = []
        for kw in test.keywords:
            rank = None
            for j, chunk in enumerate(chunks):
                if kw.lower() in chunk.page_content.lower():
                    rank = j + 1
                    break
            if rank:
                print(f"  \"{kw}\" → rank {rank}  (MRR: {1/rank:.3f})")
                mrr_scores.append(1 / rank)
            else:
                print(f"  \"{kw}\" → NOT FOUND")
                mrr_scores.append(0.0)
        print(f"  Question MRR: {sum(mrr_scores)/len(mrr_scores):.3f}\n")


def run_pipeline(name, overwrite=False):
    fetch_fn, sleep_secs = PIPELINES[name]
    tests = load_tests()
    RESULTS_DIR.mkdir(exist_ok=True)

    bi_short = BI_ENCODER.split("/")[-1]
    ce_short = CROSS_ENCODER_MODEL.split("/")[-1].replace("ms-marco-", "")
    llm_short = model.split("/")[-1]

    if name in ("cross_encoder", "hybrid"):
        filename = RESULTS_DIR / f"{name}_{bi_short}_{ce_short}_{llm_short}.json"
    else:
        filename = RESULTS_DIR / f"{name}_{bi_short}_{llm_short}.json"

    if filename.exists() and not overwrite:
        answer = input(f"\n{filename.name} already exists. Replace it? (y/n): ").strip().lower()
        if answer != "y":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = RESULTS_DIR / f"{filename.stem}_{timestamp}.json"
            print(f"Saving to {filename.name} instead.")

    print_model_info()
    print(f"Running: {name} ({len(tests)} questions)")
    if sleep_secs:
        print(f"Sleeping {sleep_secs}s between questions — est. {sleep_secs * (len(tests) - 1) // 60 + 1} min\n")

    all_mrr, all_ndcg, coverage_scores = [], [], []
    for i, test in enumerate(tests):
        print(f"  [{i+1}/{len(tests)}] {test.question[:60]}...")
        chunks = fetch_fn(test.question)
        mrr_per_kw = [calculate_mrr(kw, chunks) for kw in test.keywords]
        ndcg_per_kw = [calculate_ndcg(kw, chunks) for kw in test.keywords]
        all_mrr.append(sum(mrr_per_kw) / len(mrr_per_kw))
        all_ndcg.append(sum(ndcg_per_kw) / len(ndcg_per_kw))
        coverage_scores.append(keyword_coverage(chunks, test.keywords))
        if sleep_secs and i < len(tests) - 1:
            time.sleep(sleep_secs)

    result = {
        "pipeline": name,
        "models": {
            "bi_encoder": BI_ENCODER,
            "cross_encoder": CROSS_ENCODER_MODEL,
            "inference_llm": model,
        },
        "avg_mrr": sum(all_mrr) / len(all_mrr),
        "avg_ndcg": sum(all_ndcg) / len(all_ndcg),
        "avg_coverage": sum(coverage_scores) / len(coverage_scores),
    }
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {filename.name}")
    print(f"MRR: {result['avg_mrr']:.3f} | nDCG: {result['avg_ndcg']:.3f} | Coverage: {result['avg_coverage']:.3f}")


def compare():
    print(f"\n{'Pipeline':<20} {'MRR':>6} {'nDCG':>7} {'Coverage':>10}  {'File'}")
    print("-" * 75)
    for name in PIPELINES:
        matches = sorted(RESULTS_DIR.glob(f"{name}_*.json"))
        if matches:
            for path in matches:
                r = json.loads(path.read_text())
                print(f"{name:<20} {r['avg_mrr']:>6.3f} {r['avg_ndcg']:>7.3f} {r['avg_coverage']:>10.3f}  {path.name}")
        else:
            print(f"{name:<20} {'(not run yet)':>28}")


if __name__ == "__main__":
    args = sys.argv[1:]
    overwrite = "-y" in args
    args = [a for a in args if a != "-y"]
    cmd = " ".join(args)

    if cmd == "compare":
        compare()
    elif cmd.startswith("debug ") and cmd[6:] in PIPELINES:
        debug_pipeline(cmd[6:])
    elif cmd in PIPELINES:
        run_pipeline(cmd, overwrite=overwrite)
    else:
        print("Usage: uv run eval.py <pipeline|compare|debug <pipeline>> [-y]")
        print(f"  Pipelines: {', '.join(PIPELINES.keys())}")
        sys.exit(1)
