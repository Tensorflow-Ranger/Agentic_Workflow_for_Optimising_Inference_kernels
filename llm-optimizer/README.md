# MGPUSim LLM Kernel Optimizer

This tool automatically optimizes AMD GCN3 OpenCL kernels using an LLM feedback loop. It compiles kernels, runs them in the MGPUSim simulator, analyzes performance metrics, and iteratively improves the code.

## How It Works

The optimizer runs a closed loop for each iteration:

1. **Rewriter** — LLM generates an optimized kernel based on the previous performance bottleneck
2. **Compiler** — Compiles the kernel to HSACO binary and rebuilds the simulator binary
3. **Simulator** — Runs the FIR benchmark in MGPUSim and collects metrics
4. **Metrics Parser** — Converts simulator output to CSV and extracts key counters
5. **Profiler** — Analyzes metrics to identify the performance bottleneck (memory-bound, compute-bound, etc.)
6. **Planner** — Decides the next optimization strategy based on the bottleneck
7. **Loop** — Back to step 1 with the new strategy

Each iteration's metrics are archived to `history/metrics/metrics_iter*.csv` for comparison.

## Setup

### Requirements

- Go 1.20+
- Python 3.10+
- OpenCL compiler (ROCm or clang-ocl) for local fallback
- Groq API key (for LLM calls)

### Environment Variables

```bash
export GROQ_API_KEY=your_api_key_here
export GROQ_BASE_URL=https://api.groq.com/openai/v1
export LLM_MODEL=openai/gpt-oss-120b  # or any Groq-compatible model
export FIR_LENGTH=8                    # Filter kernel size (default 8)
export MAX_ITERATIONS=5                # Number of optimization cycles
export COMPILER_API_URL=http://40.192.96.193:8000  # Remote compiler endpoint
```

### Python Dependencies

```bash
pip install openai requests
```

## Usage

Run the optimizer:

```bash
cd llm-optimizer
python3 orchestrator.py --iterations 3
```

### Flags

- `--iterations N` — Override the iteration count
- `--dry-run` — Profile the baseline kernel once and exit
- `--skip-sim` — Skip simulator step, use cached metrics
- `--no-compile` — Skip compilation (useful when ROCm is unavailable)
- `--resume PATH` — Resume from a saved history.json file

## Output

All results are saved to `history/`:

- `history.json` — Full optimization log with metrics for each iteration
- `metrics/metrics_iter*.csv` — Per-iteration performance metrics
- `metrics/akita_iter*.sqlite3` — Raw simulator databases
- `kernels/kernel_iter*.cl` — Generated kernel source for each iteration
- `logs/run_*.log` — Detailed execution logs

## Example Run

```bash
GROQ_API_KEY=gsk_xxx python3 orchestrator.py --iterations 5
```

The optimizer will show real-time progress in the terminal and save the history to `history/history.json`. Each iteration's kernel time and improvement percentage is logged.

## Notes

- The simulator enforces strict 32-bit integer arithmetic (no size_t, ulong, or 64-bit shifts).
- If the Compiler API is unavailable, the optimizer falls back to local compilation (if ROCm is installed).
- If the simulator fails consistently, the optimizer can use synthetic metrics and continue to collect real data as fixes are applied.
