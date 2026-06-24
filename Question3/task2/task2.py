"""
Nomura Quant Challenge 5 — Task 2: Client Profitability & Spread Recommendation
[PDF] §2.3, 5 marks

Self-contained submission file. Run with:
    python task2.py                      # uses ./trade_data.csv next to the script
    python task2.py /path/to/trade.csv   # override the CSV path

Produces:
    task2_results.csv  next to this script.
"""
import os
import sys
from typing import List

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────
HORIZONS = [5, 10, 15, 20, 25, 30]
N_HORIZONS = len(HORIZONS)
CLIENT_IDS = ["A", "B", "C", "D", "E", "F"]
CLIENT_MAP = {c: f"Client{c}" for c in CLIENT_IDS}
CLIENT_MAP_INV = {v: k for k, v in CLIENT_MAP.items()}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILE = os.path.join(SCRIPT_DIR, "trade_data.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "task2_results.csv")


# ── Data loading ─────────────────────────────────────────────────────────────
_DATA: pd.DataFrame = None
_DATA_PATH: str = None


def _load_data(path: str = None) -> pd.DataFrame:
    """Read trade_data.csv once and cache it.
    Task 2 needs: Name, Side, Volume, Trade Price, M0, M5..M30."""
    global _DATA, _DATA_PATH
    if path is None:
        path = DEFAULT_DATA_FILE
    if _DATA is None or _DATA_PATH != path:
        _DATA = pd.read_csv(path)
        _DATA_PATH = path
    return _DATA


def _client_arrays(client: str):
    """Return (side, volume, tp, m0, m_tau_matrix) arrays for one client."""
    if client in CLIENT_MAP_INV:
        client = CLIENT_MAP_INV[client]
    df = _load_data()
    cdf = df[df["Name"] == client]
    side   = cdf["Side"].to_numpy(dtype=np.float64)
    volume = cdf["Volume"].to_numpy(dtype=np.float64)
    tp     = cdf["Trade Price"].to_numpy(dtype=np.float64)
    m0     = cdf["M0"].to_numpy(dtype=np.float64)
    m_tau  = cdf[[f"M{t}" for t in HORIZONS]].to_numpy(dtype=np.float64)  # (n, 6)
    return side, volume, tp, m0, m_tau


# ── Required PDF functions ────────────────────────────────────────────────────
def expected_pnl(client: str, tau: List[int]) -> dict:
    """
    [PDF] §2.4 — required signature.

    Returns:
        {
            'per_horizon': List[float]  — E[PnL] per trade at each tau (Corollary 1 / Eq. 5)
            'aggregate':   float        — E[AggPnL] per trade (Corollary 2 / Eq. 6)
        }

    Eq. 5:  PnL_j(tau) = side_j * V_j * (M_tau_j - TP_j)
            E[PnL(c, tau)] = mean over client's trades

    Eq. 6:  AggPnL_j = side_j * V_j * (1/6) * sum_i (M_{5i,j} - TP_j)
            E[AggPnL(c)] = mean over client's trades
    """
    side, volume, tp, _, m_tau_all = _client_arrays(client)
    n = len(side)
    if n == 0:
        return {"per_horizon": [0.0] * len(tau), "aggregate": 0.0}

    # Build the (n, 6) matrix for the requested tau subset
    tau_idx = [HORIZONS.index(t) for t in tau]
    m_req = m_tau_all[:, tau_idx]  # (n, len(tau))

    # Per-horizon PnL [Eq. 5]: side * V * (M_tau - TP), then mean
    pnl_matrix = side[:, None] * volume[:, None] * (m_req - tp[:, None])
    per_horizon = pnl_matrix.mean(axis=0).tolist()

    # Aggregate PnL [Eq. 6]: uniform 1/6 weight over all 6 horizons
    agg_pnl_trades = side * volume * (m_tau_all - tp[:, None]).mean(axis=1)
    aggregate = float(agg_pnl_trades.mean())

    return {"per_horizon": per_horizon, "aggregate": aggregate}


def classify_client(client: str) -> str:
    """
    [PDF] §2.4 — required signature.

    PnL is from the LP's perspective (PDF p.4 line 117).
    E[AggPnL] >= 0  =>  LP profits  =>  'profitable'
    E[AggPnL] < 0   =>  LP loses    =>  'costly'
    """
    agg = expected_pnl(client, HORIZONS)["aggregate"]
    return "profitable" if agg >= 0 else "costly"


def min_half_spread(client: str) -> float:
    """
    [PDF] §2.4 — required signature.

    Find delta* >= 0 such that E[AggPnL] >= 0 when LP quotes at M0 ± delta*.

    Derivation:
        New TP: TP' = M0 - side * delta   (buy: bid = M0 - delta; sell: ask = M0 + delta)
        AggPnL_j(delta) = side_j * V_j/6 * sum_i(M_{5i} - (M0 - side*delta))
                        = side_j * V_j/6 * sum_i(M_{5i} - M0)  +  side_j^2 * V_j * delta
                        = A_j  +  V_j * delta      (since side^2 = 1)

        E[AggPnL(delta)] = A_bar + V_bar * delta = 0
        =>  delta* = max(0, -A_bar / V_bar)
    """
    side, volume, tp, m0, m_tau = _client_arrays(client)
    n = len(side)
    if n == 0:
        return 0.0

    # A_bar: mean per-trade PnL at zero half-spread (measured from mid M0)
    a_trades = side * volume * (m_tau - m0[:, None]).mean(axis=1)
    a_bar = float(a_trades.mean())

    v_bar = float(volume.mean())
    if v_bar <= 0.0:
        return 0.0

    return max(0.0, -a_bar / v_bar)


# ── CSV output ────────────────────────────────────────────────────────────────
def generate_task2_csv(out_path: str = OUTPUT_CSV) -> str:
    """Write task2_results.csv per PDF §2.4 schema."""
    rows = []
    for cid in CLIENT_IDS:
        result = expected_pnl(cid, HORIZONS)
        delta = min_half_spread(cid)
        row = {"client": CLIENT_MAP[cid]}
        for t, val in zip(HORIZONS, result["per_horizon"]):
            row[f"τ = {t}"] = val
        row["agg_pnl"] = result["aggregate"]
        row["δ*"] = delta
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, float_format="%.6f", encoding="utf-8")
    print(f"[Task 2] CSV saved to: {out_path}")
    return out_path


def run_task2(data_path: str = None) -> str:
    """Print per-client results table and write the CSV."""
    global _DATA, _DATA_PATH
    _DATA, _DATA_PATH = None, None
    _load_data(data_path)

    print("=" * 70)
    print("TASK 2: CLIENT PROFITABILITY & SPREAD RECOMMENDATION")
    print("=" * 70)
    print(f"Loaded {len(_DATA)} trades from {_DATA_PATH}\n")

    # Expected PnL table
    h = f"{'Client':<10}" + "".join(f"{'τ='+str(t):>12}" for t in HORIZONS) + f"{'AggPnL':>12}"
    print(h)
    print("-" * len(h))
    for cid in CLIENT_IDS:
        res = expected_pnl(cid, HORIZONS)
        row = f"{CLIENT_MAP[cid]:<10}"
        row += "".join(f"{v:>12.4f}" for v in res["per_horizon"])
        row += f"{res['aggregate']:>12.4f}"
        print(row)

    # Classification and delta*
    print("\n--- Classification & Minimum Half-Spread ---")
    for cid in CLIENT_IDS:
        cls = classify_client(cid)
        delta = min_half_spread(cid)
        agg = expected_pnl(cid, HORIZONS)["aggregate"]
        print(f"  {CLIENT_MAP[cid]:<10} {cls:<12}  agg_pnl={agg:.4f}  δ*={delta:.6f}")

    print()
    return generate_task2_csv()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_task2(path)
