"""
Evaluation script using Gemini Batch API for LLM-as-judge evaluation.
"""

import argparse
import re
import time
import requests
import os

from dataclasses import dataclass, field
from dotenv import load_dotenv
from tqdm import tqdm

from auto_evaluation.src.metrics.prompts import METRIC_REGISTRY
from auto_evaluation.src.metrics.retrieval import RETRIEVAL_METRICS
from auto_evaluation.src.models.gemini import submit_gemini_batch, poll_gemini_batch
from auto_evaluation.dataset import hf_pull, preprocess

eval_root_path = os.path.join(os.path.dirname(__file__), "..")
load_dotenv(dotenv_path=os.path.join(eval_root_path, ".env"))

ALL_RETRIEVERS = {
    "agent-retriever": "/conversations/agent-retriever",
    "agent-retriever-reranker": "/conversations/agent-retriever",
}
RETRY_INTERVAL = 5
RETRY_TIMEOUT = 600
DEFAULT_MODEL = "gemini-2.5-pro"
THRESHOLD = 0.7


@dataclass
class EvalSample:
    question: str
    response: str
    expected: str
    contexts: list[str] = field(default_factory=list)


@dataclass
class MetricResult:
    scores: list[float]
    threshold: float = THRESHOLD

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def pass_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s >= self.threshold) / len(self.scores)


def _render_prompt(metric_name: str, sample: EvalSample) -> str:
    """Render a metric prompt template for a single EvalSample."""
    config = METRIC_REGISTRY[metric_name]
    context_str = "\n".join(f"- {c}" for c in sample.contexts)
    numbered_contexts = "\n".join(
        f"[{i + 1}] {c}" for i, c in enumerate(sample.contexts)
    )
    return str(config["template"]).format(
        question=sample.question,
        response=sample.response,
        expected=sample.expected,
        context=context_str,
        numbered_contexts=numbered_contexts,
    )


def _parse_score(metric_name: str, response_text: str) -> float:
    """Parse a Gemini response into a float score [0.0, 1.0]."""
    config = METRIC_REGISTRY[metric_name]
    text = response_text.strip().lower()

    if config["response_type"] == "binary":
        rails: dict[str, float] = config["rails"]
        # Match longest label first to avoid partial matches (e.g. "not toxic" vs "toxic")
        for label in sorted(rails, key=len, reverse=True):
            if label in text:
                return rails[label]
        return 0.0

    # Numeric: extract the first float in [0.0, 1.0]
    matches = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if matches:
        return float(matches[0])
    return 0.0


class BatchGeminiEvaluator:
    """
    Runs LLM-as-judge evaluations via Google Gemini Batch API.

    Usage:
        evaluator = BatchGeminiEvaluator()
        results = evaluator.evaluate(samples, metrics=["hallucination", "contextual_precision"])
        for metric, result in results.items():
            print(metric, result.mean, result.pass_rate)
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def evaluate(
        self,
        samples: list[EvalSample],
        metrics: list[str] | None = None,
    ) -> dict[str, MetricResult]:
        if not samples:
            return {}
        if metrics is None:
            metrics = RETRIEVAL_METRICS

        for m in metrics:
            if m not in METRIC_REGISTRY:
                raise ValueError(
                    f"Unknown metric '{m}'. Available: {list(METRIC_REGISTRY)}"
                )

        # Phase 1: render all prompts, preserving insertion order for positional mapping
        ordered_keys: list[tuple[str, int]] = []  # (metric_name, sample_idx)
        prompts: list[str] = []
        for metric_name in metrics:
            for i, sample in enumerate(samples):
                prompts.append(_render_prompt(metric_name, sample))
                ordered_keys.append((metric_name, i))

        print(
            f"Phase 1 complete: {len(prompts)} prompts rendered "
            f"({len(samples)} samples × {len(metrics)} metrics)"
        )

        # Phase 2: submit and poll Gemini Batch job (responses positionally aligned)
        job_name = submit_gemini_batch(prompts, model=self.model)
        responses = poll_gemini_batch(job_name)

        # Phase 3: parse scores and aggregate
        raw: dict[str, list[tuple[int, float]]] = {m: [] for m in metrics}
        for pos, response_text in enumerate(responses):
            metric_name, idx = ordered_keys[pos]
            score = _parse_score(metric_name, response_text)
            raw[metric_name].append((idx, score))

        results: dict[str, MetricResult] = {}
        for metric_name in metrics:
            sorted_scores = [s for _, s in sorted(raw[metric_name])]
            results[metric_name] = MetricResult(
                scores=sorted_scores, threshold=THRESHOLD
            )

        _print_results(results)
        return results


def _print_results(results: dict[str, MetricResult]) -> None:
    print("\n--- Evaluation Results ---")
    print(f"{'Metric':<30} {'Mean Score':>12} {'Pass Rate':>12}")
    print("-" * 56)
    for metric_name, result in results.items():
        print(
            f"{metric_name:<30} {result.mean:>12.4f} {result.pass_rate:>11.1%}"
        )
    print()


class EvaluationHarness:
    def __init__(self, base_url: str, dataset: str, reranker_base_url: str = ""):
        self.base_url = base_url
        self.dataset = dataset
        self.reranker_base_url = reranker_base_url
        self.qns = preprocess.read_data(self.dataset)
        self.evaluator = BatchGeminiEvaluator(model=DEFAULT_MODEL)
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.sanity_check()

    def sanity_check(self):
        cur_time = time.time()
        if not os.path.exists(self.dataset):
            raise ValueError("Dataset path does not exist")
        while time.time() - cur_time < RETRY_TIMEOUT:
            try:
                if not requests.get(f"{self.base_url}/healthcheck").status_code == 200:
                    print("Endpoint not ready, retrying...")
                    time.sleep(RETRY_INTERVAL)
                    continue
                if self.reranker_base_url and not (
                    requests.get(
                        f"{self.reranker_base_url}/healthcheck"
                    ).status_code
                    == 200
                ):
                    print("Reranker endpoint not ready, retrying...")
                    time.sleep(RETRY_INTERVAL)
                    continue
                return
            except requests.exceptions.RequestException:
                print("Connection failed, retrying...")
                time.sleep(RETRY_INTERVAL)
                continue
        raise ValueError("Sanity check failed after timeout")

    def evaluate(
        self,
        retriever: str,
        limit: int | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, MetricResult]:
        samples: list[EvalSample] = []
        questions = self.qns[:limit] if limit else self.qns

        for qa_pair in tqdm(questions, desc="Querying retriever"):
            question, ground_truth = qa_pair["question"], qa_pair["ground_truth"]
            response, _ = self.query(retriever, question)
            samples.append(
                EvalSample(
                    question=question,
                    response=response["response"],
                    expected=ground_truth,
                    contexts=[r["context"] for r in response["context_sources"]],
                )
            )

        return self.evaluator.evaluate(samples, metrics=metrics)

    def query(self, retriever: str, query: str) -> tuple[dict, float]:
        endpoint = ALL_RETRIEVERS[retriever]
        url = (
            f"{self.base_url}/{endpoint}"
            if retriever != "agent-retriever-reranker"
            else f"{self.reranker_base_url}/{endpoint}"
        )
        payload = {"query": query, "list_context": True, "list_sources": False}
        try:
            time.sleep(5)
            response = requests.post(url, json=payload)
            if not response.ok:
                print(f"Error querying {retriever}: HTTP {response.status_code}")
                return (
                    {"response": "invalid", "context_sources": [], "tools": []},
                    -999999,
                )
            return response.json(), response.elapsed.total_seconds() * 1000
        except Exception as e:
            print(f"Error querying {retriever}: {e}")
            return (
                {"response": "invalid", "context_sources": [], "tools": []},
                -999999,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--base_url", type=str, help="Base URL of the model to evaluate")
    parser.add_argument(
        "--reranker_base_url", type=str, help="Base URL of the reranker", default=""
    )
    parser.add_argument("--dataset", type=str, help="Path to dataset to evaluate on")
    parser.add_argument("--retriever", type=str, help="Retriever to evaluate on")
    parser.add_argument(
        "--limit", type=int, help="Limit number of questions to evaluate", default=None
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help=f"Metrics to evaluate. Available: {list(METRIC_REGISTRY)}",
    )
    args = parser.parse_args()

    hf_pull.main()

    harness = EvaluationHarness(args.base_url, args.dataset, args.reranker_base_url)
    harness.evaluate(args.retriever, limit=args.limit, metrics=args.metrics)
