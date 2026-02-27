"""
Agent 1 — Rewriter
==================
Takes the current kernel source + a strategy instruction from the Planner,
and returns an optimised OpenCL kernel string ready to compile.
"""

from pathlib import Path
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set.\n"
                "Export it with:  export GROQ_API_KEY=<your-key>"
            )
        _client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
        )
    return _client


SYSTEM_PROMPT = f"""You are an expert GPU compute engineer specialising in AMD GCN3 OpenCL kernels.

Your sole job is to rewrite a given OpenCL kernel so it runs faster on AMD GCN3 hardware.
You are given:
  1. The current kernel source code.
  2. A concrete optimisation strategy instruction from the Planning Agent.

Output rules (STRICT):
  - Return ONLY the complete, compilable OpenCL kernel source code.
  - Do NOT include markdown code fences (```), explanations, or any text outside the kernel.
  - Preserve the exact kernel function name and argument signature.
  - You may add helper functions or preprocessor macros before the kernel.

{config.GCN3_HINTS}

Optimisation techniques you know:
  - LDS (Local Data Store) tiling: load coeff[] into __local before the loop
  - Loop unrolling with #pragma unroll
  - Vectorised loads using float4 / float8
  - Avoiding branch divergence by splitting the prologue separately
  - Increasing wavefront occupancy by reducing VGPR pressure
  - Software prefetching patterns
  - Coalesced memory access via work-group tiling
"""


def rewrite(kernel_source: str, strategy: str) -> str:
    """
    Call the LLM to rewrite kernel_source according to strategy.
    Returns the new kernel source code as a string.
    """
    user_message = (
        f"## Current kernel\n\n{kernel_source}\n\n"
        f"## Optimisation instruction\n\n{strategy}\n\n"
        "Return the optimised kernel now."
    )

    response = _get_client().chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    return response.choices[0].message.content.strip()
