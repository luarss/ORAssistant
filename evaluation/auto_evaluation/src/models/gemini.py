"""
Gemini Batch API utilities for evaluation.

Submits all LLM judge calls as a single Gemini Batch job (50% cheaper,
async, non-blocking) instead of making real-time synchronous calls.

Responses are returned positionally — the caller is responsible for
maintaining the ordered list of keys that maps to each prompt.
"""

import time
import os

from google import genai
from google.genai import types


_COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def _make_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def submit_gemini_batch(
    prompts: list[str],
    model: str = "gemini-2.5-pro",
    display_name: str = "ora-eval-job",
) -> str:
    """
    Submit an ordered list of prompt strings to Gemini Batch API.

    Returns the batch job name. Responses will be returned positionally
    by poll_gemini_batch(), in the same order as the input prompts.
    """
    client = _make_client()
    src = types.BatchJobSource(
        inlined_requests=[
            types.InlinedRequest(contents=prompt) for prompt in prompts
        ]
    )
    batch_job = client.batches.create(
        model=model,
        src=src,
        config=types.CreateBatchJobConfig(display_name=display_name),
    )
    job_name: str = str(batch_job.name)
    print(f"Submitted Gemini batch job: {job_name} ({len(prompts)} requests)")
    return job_name


def poll_gemini_batch(
    job_name: str,
    poll_interval: int = 30,
) -> list[str]:
    """
    Poll a Gemini Batch job until completion.

    Returns an ordered list of response text strings, one per input prompt.
    Raises RuntimeError if the job fails, is cancelled, or expires.
    """
    client = _make_client()

    while True:
        batch_job = client.batches.get(name=job_name)
        if batch_job is None:
            raise RuntimeError(f"Gemini batch job {job_name} not found")

        state_obj = batch_job.state
        state: str = state_obj.name if state_obj is not None else "UNKNOWN"
        print(f"Batch job state: {state}")

        if state not in _COMPLETED_STATES:
            time.sleep(poll_interval)
            continue

        if state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(
                f"Gemini batch job {job_name} ended with state {state}: "
                f"{batch_job.error}"
            )

        results: list[str] = []
        dest = batch_job.dest
        if dest is None or dest.inlined_responses is None:
            return results

        for inline_response in dest.inlined_responses:
            if inline_response.error is not None:
                print(f"Warning: a request failed: {inline_response.error}")
                results.append("")
                continue

            text = ""
            resp = inline_response.response
            if resp is not None:
                try:
                    raw = resp.text
                    text = str(raw) if raw is not None else ""
                except AttributeError:
                    candidate = getattr(resp, "candidates", [None])[0]
                    content = getattr(candidate, "content", None)
                    part = getattr(content, "parts", [None])[0]
                    raw = getattr(part, "text", None)
                    text = str(raw) if raw is not None else ""
            results.append(text)

        return results
