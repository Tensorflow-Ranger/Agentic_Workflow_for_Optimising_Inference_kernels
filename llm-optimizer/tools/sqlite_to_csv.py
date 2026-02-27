"""
sqlite_to_csv.py — Convert MGPUSim's akita_sim_*.sqlite3 output to metrics.csv

The MGPUSim simulator writes a SQLite database with the schema:
    mgpusim_metrics(Location TEXT, What TEXT, Value REAL, Unit TEXT)

This module provides:
    find_latest_sqlite(directory, min_mtime=None)  → Path | None
    sqlite_to_csv(db_path, csv_path)               → Path
"""

import csv
import sqlite3
import time
from pathlib import Path


def find_latest_sqlite(directory: Path | str, min_mtime: float | None = None) -> "Path | None":
    """
    Return the most recently modified akita_sim_*.sqlite3 file in directory.

    Parameters
    ----------
    directory : path to search (non-recursive)
    min_mtime : if given, only files with mtime >= min_mtime are considered.
                Pass time.time() captured just before launching the simulator
                to ensure you never pick up a stale database from a prior run.

    Returns None if no matching file is found.
    """
    directory = Path(directory)
    candidates = list(directory.glob("akita_sim_*.sqlite3"))

    if min_mtime is not None:
        candidates = [p for p in candidates if p.stat().st_mtime >= min_mtime]

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def sqlite_to_csv(db_path: Path | str, csv_path: Path | str) -> Path:
    """
    Read all rows from mgpusim_metrics and write them to a CSV file.

    CSV columns: Location, What, Value, Unit

    Returns the path to the written CSV.
    """
    db_path  = Path(db_path)
    csv_path = Path(csv_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT Location, What, Value, Unit FROM mgpusim_metrics ORDER BY rowid"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Location", "What", "Value", "Unit"])
        writer.writerows(rows)

    return csv_path
