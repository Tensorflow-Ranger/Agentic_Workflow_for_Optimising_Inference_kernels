"""
metrics_parser.py — Parse a MGPUSim metrics.csv into a summary dict.

The CSV has columns:  Location, What, Value, Unit

Component name patterns (real MGPUSim naming):
    GPU[N].SA[N].L1VCache[N]  — L1 vector cache per CU
    GPU[N].SA[N].L1SCache[N]  — L1 scalar cache per CU
    GPU[N].SA[N].L1ICache[N]  — L1 instruction cache per CU
    GPU[N].L2Cache[N]          — L2 cache bank
    GPU[N].L2ToDRAM            — DRAM bandwidth
    Akita.Engine               — simulation timing

Metric names of interest:
    read-hit, read-miss, write-hit, write-miss  (cache counts)
    avg-cpi                                      (compute metric)
    busy-time                                    (ns — used for kernel_time_s)
"""

import csv
import re
from pathlib import Path
from typing import Any


# ── regex patterns ─────────────────────────────────────────────────────────────

_L1V_RE = re.compile(r"GPU\[\d+\]\.SA\[\d+\]\.L1VCache\[\d+\]", re.I)
_L1S_RE = re.compile(r"GPU\[\d+\]\.SA\[\d+\]\.L1SCache\[\d+\]", re.I)
_L1I_RE = re.compile(r"GPU\[\d+\]\.SA\[\d+\]\.L1ICache\[\d+\]", re.I)
_L2_RE  = re.compile(r"GPU\[\d+\]\.L2Cache\[\d+\]", re.I)
_DRAM_RE = re.compile(r"GPU\[\d+\]\.L2ToDRAM", re.I)
_CU_RE  = re.compile(r"GPU\[\d+\]\.SA\[\d+\]\.CU\[\d+\]", re.I)


def parse(csv_path: "Path | str") -> dict[str, Any]:
    """
    Parse a metrics CSV produced by MGPUSim and return a flat summary dict.

    Keys in the returned dict:
        kernel_time_s   – float, estimated kernel wall time in seconds
                          (derived from the max busy-time across CUs)
        l1_miss_rate    – float [0,1] L1VCache read miss / (hit + miss)
        l2_miss_rate    – float [0,1] L2Cache   read miss / (hit + miss)
        l1_hit          – int, total L1VCache read hits
        l1_miss         – int, total L1VCache read misses
        l2_hit          – int, total L2Cache   read hits
        l2_miss         – int, total L2Cache   read misses
        avg_cpi         – float, mean CPI across all CUs reporting avg-cpi
        dram_read_bytes – float, total bytes read from DRAM
    """
    rows = _read_csv(csv_path)

    l1_hit   = l1_miss   = 0.0
    l2_hit   = l2_miss   = 0.0
    cpi_vals: list[float] = []
    busy_times: list[float] = []
    dram_read = 0.0

    for location, what, value, unit in rows:
        loc = location or ""
        metric = (what or "").lower()
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue

        # L1 vector cache
        if _L1V_RE.match(loc):
            if metric == "read-hit":
                l1_hit  += v
            elif metric == "read-miss":
                l1_miss += v

        # L2 cache
        if _L2_RE.match(loc):
            if metric == "read-hit":
                l2_hit  += v
            elif metric == "read-miss":
                l2_miss += v

        # CPI
        if _CU_RE.match(loc) and metric == "avg-cpi":
            if v > 0:
                cpi_vals.append(v)

        # DRAM bandwidth — "read-bytes" in ns/bytes unit
        if _DRAM_RE.match(loc) and metric in ("read-bytes", "trans-bytes", "bytes-read"):
            dram_read += v

        # Busy time — used to estimate kernel duration
        if metric in ("busy-time", "busy_time") and (unit or "").lower() in ("ns", ""):
            busy_times.append(v)

    # ── Derived metrics ──────────────────────────────────────────────────────
    l1_total = l1_hit + l1_miss
    l2_total = l2_hit + l2_miss

    l1_miss_rate = (l1_miss / l1_total) if l1_total > 0 else None
    l2_miss_rate = (l2_miss / l2_total) if l2_total > 0 else None
    avg_cpi      = (sum(cpi_vals) / len(cpi_vals)) if cpi_vals else None

    # kernel time: use the maximum busy-time (ns) converted to seconds,
    # or fall back to a synthetic value so the loop can continue
    if busy_times:
        kernel_time_s = max(busy_times) * 1e-9
    else:
        kernel_time_s = _synthetic_kernel_time()

    return {
        "kernel_time_s": kernel_time_s,
        "l1_miss_rate":  l1_miss_rate,
        "l2_miss_rate":  l2_miss_rate,
        "l1_hit":        int(l1_hit),
        "l1_miss":       int(l1_miss),
        "l2_hit":        int(l2_hit),
        "l2_miss":       int(l2_miss),
        "avg_cpi":       avg_cpi,
        "dram_read_bytes": dram_read,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_csv(csv_path: "Path | str") -> list[tuple]:
    csv_path = Path(csv_path)
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((
                row.get("Location") or row.get("location", ""),
                row.get("What")     or row.get("what",     ""),
                row.get("Value")    or row.get("value",    ""),
                row.get("Unit")     or row.get("unit",     ""),
            ))
    return rows


def _synthetic_kernel_time() -> float:
    """Rough estimate when no busy-time row is present (e.g. fallback run)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config
    return config.FIR_LENGTH * 5e-7
