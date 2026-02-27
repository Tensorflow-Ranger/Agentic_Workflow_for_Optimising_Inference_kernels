"""
Agent 2 — Profiler
==================
Interprets the parsed metrics dict and returns a bottleneck JSON describing
what is limiting kernel performance and why.

The output of interpret() is fed directly to the Planner.
"""

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set.")
        _client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
        )
    return _client


SYSTEM_PROMPT = f"""You are an expert GPU performance analyst for AMD GCN3 hardware.

You receive a JSON dict of hardware performance counters collected from MGPUSim.
Your job: identify the primary performance bottleneck and explain it concisely.

Output ONLY a JSON object with exactly these keys:
  "bound_by"           : one of "memory_bound" | "compute_bound" | "latency_bound" |
                          "underutilised" | "sim_panic" | "unknown"
  "bottleneck_summary" : 1-2 sentence plain-English explanation
  "l1_miss_rate"       : the l1_miss_rate value from input (float or null)
  "l2_miss_rate"       : the l2_miss_rate value from input (float or null)
  "avg_cpi"            : the avg_cpi value from input (float or null)
  "kernel_time_s"      : the kernel_time_s value from input (float or null)
  "recommendations"    : list of 2-4 short actionable strings

Decision heuristics:
  - l1_miss_rate > 0.3  OR  l2_miss_rate > 0.2  → memory_bound
  - avg_cpi < 2.0  AND  l1_miss_rate < 0.15      → compute_bound
  - avg_cpi > 6.0                                  → latency_bound
  - kernel_time_s very small AND low miss rates    → underutilised (too few threads)
  - When in doubt: memory_bound (most common for stencil/FIR kernels)

{config.GCN3_HINTS}
"""


def interpret(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Given a metrics dict from metrics_parser.parse(), return a bottleneck dict.

    Falls back to a heuristic-only answer if the LLM call fails.
    """
    try:
        return _llm_interpret(metrics)
    except Exception as exc:
        return _heuristic_interpret(metrics, str(exc))


def _llm_interpret(metrics: dict[str, Any]) -> dict[str, Any]:
    user_message = (
        f"## Hardware performance metrics\n\n"
        f"{json.dumps(metrics, indent=2, default=str)}\n\n"
        "Identify the bottleneck and return the JSON object now."
    )

    response = _get_client().chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()
    result = json.loads(raw)

    # Ensure required keys exist
    result.setdefault("bound_by",           "unknown")
    result.setdefault("bottleneck_summary", "")
    result.setdefault("recommendations",    [])
    return result


def _heuristic_interpret(metrics: dict[str, Any], error: str) -> dict[str, Any]:
    """Simple rule-based fallback when the LLM is unavailable."""
    l1 = metrics.get("l1_miss_rate")
    l2 = metrics.get("l2_miss_rate")
    cpi = metrics.get("avg_cpi")

    if l1 is not None and l1 > 0.3:
        bound = "memory_bound"
        summary = f"L1 miss rate is {l1:.1%} — kernel is memory bound."
    elif l2 is not None and l2 > 0.2:
        bound = "memory_bound"
        summary = f"L2 miss rate is {l2:.1%} — kernel is memory bound."
    elif cpi is not None and cpi > 6.0:
        bound = "latency_bound"
        summary = f"Average CPI is {cpi:.1f} — kernel is latency bound."
    elif cpi is not None and cpi < 2.0:
        bound = "compute_bound"
        summary = f"Average CPI is {cpi:.1f} — kernel may be compute bound."
    else:
        bound = "unknown"
        summary = f"Cannot determine bottleneck from available metrics. LLM error: {error}"

    return {
        "bound_by":           bound,
        "bottleneck_summary": summary,
        "l1_miss_rate":       l1,
        "l2_miss_rate":       l2,
        "avg_cpi":            cpi,
        "kernel_time_s":      metrics.get("kernel_time_s"),
        "recommendations":    ["Use LDS tiling for filter coefficients.",
                               "Try 256 work-items per work-group."],
    }
