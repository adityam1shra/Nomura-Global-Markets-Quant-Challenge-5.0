"""
Nomura Quant Challenge 5 — Task 1: Adversity Profile
[PDF] §2.1, 5 marks

Self-contained submission file. Run with:
    python task1.py                      # uses ./trade_data.csv next to the script
    python task1.py /path/to/trade.csv   # override the CSV path

Produces:
    task1_results.csv  next to this script.
"""
import os
import sys
from typing import List

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────
HORIZONS = [5, 10, 15, 20, 25, 30]
CLIENT_IDS = ["A", "B", "C", "D", "E", "F"]
CLIENT_MAP = {c: f"Client{c}" for c in CLIENT_IDS}
CLIENT_MAP_INV = {v: k for k, v in CLIENT_MAP.items()}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILE = os.path.join(SCRIPT_DIR, "trade_data.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "task1_results.csv")


# ── Data loading ─────────────────────────────────────────────────────────────
_DATA: pd.DataFrame = None
_DATA_PATH: str = None


def _load_data(path: str = None) -> pd.DataFrame:
    """Read trade_data.csv once and cache it. Task 1 only needs Name, Side,
    Trade Price, and the M{tau} columns — no rolling vol, no features."""
    global _DATA, _DATA_PATH
    if path is None:
        path = DEFAULT_DATA_FILE
    if _DATA is None or _DATA_PATH != path:
        _DATA = pd.read_csv(path)
        _DATA_PATH = path
    return _DATA


# ── Required PDF function ────────────────────────────────────────────────────
def adversity_profile(client: str, tau: List[int]) -> List[float]:
    """
    [PDF] §2.2 — required signature.

    Returns the adversity percentage (0-100) for ``client`` at each horizon
    in ``tau``. A trade is adverse iff  side * V * (M_tau - TP) < 0
    (PDF Def 2.4, strict inequality; V > 0 always, so it cancels in the
    sign test and PnL = 0 is NOT counted as adverse).

    ``client`` may be passed as the raw single-letter id ('A') or as the
    display name ('ClientA').
    """
    if client in CLIENT_MAP_INV:
        client = CLIENT_MAP_INV[client]

    df = _load_data()
    cdf = df[df["Name"] == client]
    n = len(cdf)
    if n == 0:
        return [0.0] * len(tau)

    side = cdf["Side"].to_numpy(dtype=np.int8)
    tp = cdf["Trade Price"].to_numpy(dtype=np.float64)
    m_tau = cdf[[f"M{t}" for t in tau]].to_numpy(dtype=np.float64)

    adverse = (side[:, None] * (m_tau - tp[:, None])) < 0
    return (adverse.mean(axis=0) * 100.0).tolist()


# ── CSV output ───────────────────────────────────────────────────────────────
def generate_task1_csv(out_path: str = OUTPUT_CSV) -> str:
    """Write task1_results.csv per PDF §2.2 schema."""
    rows = []
    for cid in CLIENT_IDS:
        adv = adversity_profile(cid, HORIZONS)
        row = {"client": CLIENT_MAP[cid]}
        for t, val in zip(HORIZONS, adv):
            row[f"τ = {t}"] = val
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, float_format="%.4f", encoding="utf-8")
    print(f"[Task 1] CSV saved to: {out_path}")
    return out_path


def run_task1(data_path: str = None) -> str:
    """Print the adversity table and write the CSV."""
    global _DATA, _DATA_PATH
    _DATA, _DATA_PATH = None, None  # force a fresh load when a path is given
    df = _load_data(data_path)

    print("=" * 70)
    print("TASK 1: ADVERSITY PROFILE")
    print("=" * 70)
    print(f"Loaded {len(df)} trades from {_DATA_PATH}")

    header = f"{'Client':<10}" + "".join(f"{'τ='+str(t):>10}" for t in HORIZONS)
    print(header)
    print("-" * len(header))
    for cid in CLIENT_IDS:
        adv = adversity_profile(cid, HORIZONS)
        print(f"{CLIENT_MAP[cid]:<10}" + "".join(f"{v:>10.4f}" for v in adv))

    return generate_task1_csv()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_task1(path)
