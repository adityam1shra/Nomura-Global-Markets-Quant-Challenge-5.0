"""
Nomura Quant Challenge 5 — Task 4: Optimal Externalization Threshold
[PDF] §3.4, 15 marks

Self-contained submission file. Run with:
    python task4.py                      # uses ./trade_data.csv next to the script
    python task4.py /path/to/trade.csv   # override CSV path

Depends on:
    task3_model.pkl  (created by task3.py — must be in the same directory)

Produces:
    task4_results.csv     — per (client, τ) optimal θ* and test PnL
    pnl_vs_theta.png      — per-client PnL_validation(θ) for τ ∈ {5,10,15,20,25,30}

Strategy (PDF §3.1):
    Externalize iff p > θ (strict). Externalized trades contribute 0 PnL.
    PnL of internalized trade = side · V · (M_τ − TP)  (Eq. 5)

Design notes:
    1. Per-client θ*: one threshold per (client, τ) pair, matching the PDF §3.5 schema.
    2. Unseen-client fallback: if a client has no validation trades for a given τ,
       the global θ* (over all val trades at that τ) is used instead of a constant.
    3. Smoothed argmax: the PnL-vs-θ curve is averaged with a narrow uniform
       filter (width _SMOOTH_W = 5 steps = 0.05 in θ) before the argmax is taken.
       A genuine broad peak is barely affected; a narrow spike produced by a tiny
       slice of "very low-risk" val trades is damped to near-zero and loses to
       θ=0. No client-specific hyperparameter is involved.
    4. Plot: per-client PnL curves with individual θ* markers, consistent with
       the CSV output and directly supporting the writeup.
"""
import os
import sys
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression  # noqa: F401  (needed for unpickling)
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401


# ── Constants ────────────────────────────────────────────────────────────────
HORIZONS    = [5, 10, 15, 20, 25, 30]
N_HORIZONS  = len(HORIZONS)
CLIENT_IDS  = ["A", "B", "C", "D", "E", "F"]
CLIENT_MAP  = {c: f"Client{c}" for c in CLIENT_IDS}
TRAIN_DAYS  = 25
VAL_DAYS    = 8
VOL_WINDOW_N = 20
THETA_GRID  = np.round(np.arange(0.0, 1.005, 0.01), 4)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "trade_data.csv")
MODEL_PATH  = os.path.join(SCRIPT_DIR, "task3_model.pkl")
OUTPUT_CSV  = os.path.join(SCRIPT_DIR, "task4_results.csv")
PLOT_PATH   = os.path.join(SCRIPT_DIR, "pnl_vs_theta.png")

T_OPEN_SEC   = 9 * 3600 + 30 * 60
TRADING_SECS = (16 * 60 - (9 * 60 + 30)) * 60

# Uniform-filter width applied to the PnL-vs-θ curve before argmax.
# Averaging 5 consecutive grid points (span = 0.05 in θ) damps narrow spikes
# without shifting the argmax of a genuine broad peak.
_SMOOTH_W = 5

# Matplotlib tab10 palette (first 6 entries) — one colour per client in the plot.
_CLIENT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ═══════════════════════════════════════════════════════════════════════════════
# Inline feature pipeline (mirrors task3.py exactly so model inputs match)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_raw(path: str = None) -> pd.DataFrame:
    if path is None:
        path = DEFAULT_DATA
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["datetime"] = pd.to_datetime(
        df["Date"].dt.strftime("%Y-%m-%d") + " " + df["time"]
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    unique_dates = sorted(df["Date"].dt.date.unique())
    date_to_day = {d: i + 1 for i, d in enumerate(unique_dates)}
    df["day_num"] = df["Date"].dt.date.map(date_to_day)
    return df


def _rolling_volatility(df: pd.DataFrame) -> np.ndarray:
    N = VOL_WINDOW_N
    m0 = df["M0"].values
    day_nums = df["day_num"].values
    n = len(m0)

    returns = np.zeros(n)
    for i in range(1, n):
        if day_nums[i] != day_nums[i - 1]:
            returns[i] = 0.0
        elif m0[i - 1] != 0.0:
            returns[i] = (m0[i] - m0[i - 1]) / m0[i - 1]

    r2 = returns ** 2
    sigma = np.zeros(n)
    for i in range(1, n):
        window = r2[max(1, i - N): i]
        sigma[i] = np.sqrt(window.mean()) if len(window) > 0 else 0.0

    sigma = np.where(np.isnan(sigma), 1e-10, sigma)
    return np.maximum(sigma, 1e-10)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    client_to_id = {c: i for i, c in enumerate(CLIENT_IDS)}
    df["client_id"]    = df["Name"].map(client_to_id).astype(int)
    df["side"]         = df["Side"].astype(float)
    df["volume"]       = df["Volume"].astype(float)
    df["log_volume"]   = np.log(df["Volume"].astype(float))
    df["spread"]       = df["Spread"].astype(float)
    df["hour"]         = df["datetime"].dt.hour.astype(float)
    df["minute_of_day"]= (df["datetime"].dt.hour * 60 + df["datetime"].dt.minute).astype(float)
    df["day_of_week"]  = df["datetime"].dt.dayofweek.astype(float)

    secs = (df["datetime"].dt.hour * 3600
            + df["datetime"].dt.minute * 60
            + df["datetime"].dt.second)
    df["eta"] = np.clip((secs - T_OPEN_SEC) / TRADING_SECS, 0.0, 1.0)

    df["rolling_vol"] = _rolling_volatility(df)
    df["vol_spread"]  = df["volume"] * df["spread"]

    df["adv5_label"] = ((df["Side"] * (df["M5"] - df["Trade Price"])) < 0).astype(float)
    df["rolling_adv"] = (
        df.groupby("Name")["adv5_label"]
          .transform(lambda s: s.shift(1).rolling(50, min_periods=1).mean())
    ).fillna(0.45)

    df["signed_vol"] = df["Side"] * df["Volume"]
    df["order_flow"] = (
        df.groupby(["Name", "day_num"])["signed_vol"]
          .transform(lambda s: s.cumsum() - s)
    ).astype(float)

    df["spread_zscore"] = (
        df.groupby("Name")["spread"]
          .transform(lambda s: (s - s.shift(1).rolling(50, min_periods=5).mean())
                               / (s.shift(1).rolling(50, min_periods=5).std() + 1e-8))
    ).fillna(0.0)

    df["m0_lag5"] = (
        df.groupby(["Name", "day_num"])["M0"]
          .transform(lambda s: s.shift(5))
    )
    df["mom_short"] = ((df["M0"] - df["m0_lag5"]) / (df["m0_lag5"] + 1e-8)).fillna(0.0)

    # PnL columns: required by Task 4 (not Task 3)
    side, vol, tp = df["Side"].values, df["Volume"].values, df["Trade Price"].values
    for tau in HORIZONS:
        df[f"pnl_{tau}"] = side * vol * (df[f"M{tau}"].values - tp)

    return df


def _split_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["day_num"] <= TRAIN_DAYS].copy()
    val   = df[(df["day_num"] > TRAIN_DAYS) & (df["day_num"] <= TRAIN_DAYS + VAL_DAYS)].copy()
    test  = df[df["day_num"] > TRAIN_DAYS + VAL_DAYS].copy()
    return train, val, test


BASE_COLS = [
    "client_id", "side", "volume", "log_volume", "spread",
    "hour", "minute_of_day", "day_of_week", "eta", "rolling_vol",
    "vol_spread", "rolling_adv", "order_flow", "spread_zscore", "mom_short",
]


def _build_matrix_tau(df: pd.DataFrame, tau: int) -> np.ndarray:
    """Build single-τ feature matrix matching task3.py's training layout."""
    return np.column_stack([df[BASE_COLS].values, np.full(len(df), float(tau))]).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading (mirrors task3.py persistence format)
# ═══════════════════════════════════════════════════════════════════════════════

class _IsotonicCalibrator:
    """Mirror of task3.py's calibrator so the pickle reconstructs cleanly."""

    def __init__(self, base):
        self.base = base
        self.iso  = IsotonicRegression(out_of_bounds="clip")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.iso.transform(raw)
        return np.column_stack([1.0 - cal, cal])


def _load_model() -> _IsotonicCalibrator:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run task3.py first to create it."
        )
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    cal = _IsotonicCalibrator(data["hgbt"])
    cal.iso = data["iso"]
    return cal


# ═══════════════════════════════════════════════════════════════════════════════
# Externalization strategy & PnL sweep
# ═══════════════════════════════════════════════════════════════════════════════

def _sweep_pnl(predictions: np.ndarray, pnl: np.ndarray,
               theta_grid: np.ndarray = THETA_GRID) -> np.ndarray:
    """
    Total internalized PnL at each θ in theta_grid.

    Strategy (PDF §3.1): externalize iff p > θ. Internalized iff p ≤ θ.
    Vectorised via sort + cumulative sum: O(N log N) overall instead of
    O(|grid| · N).
    """
    order  = np.argsort(predictions)
    p_srt  = predictions[order]
    pnl_srt = pnl[order]
    cum   = np.concatenate(([0.0], np.cumsum(pnl_srt)))   # cum[k] = sum of first k sorted pnl
    # For each θ, count how many sorted preds are ≤ θ — that's the internalized count.
    counts = np.searchsorted(p_srt, theta_grid, side="right")
    return cum[counts]


def _argmax_theta(predictions: np.ndarray, pnl: np.ndarray,
                  theta_grid: np.ndarray = THETA_GRID) -> Tuple[float, np.ndarray]:
    """
    Return (θ*, PnL-curve).

    θ* is chosen as the argmax of the PnL curve after a mild uniform-filter
    smoothing (width _SMOOTH_W grid steps).  A genuine broad peak is unaffected;
    a narrow spike caused by a tiny fraction of very-low-risk val trades is damped
    and loses to the flat zero at θ=0, preventing spurious over-externalisation.
    The returned curve is the original (unsmoothed) values — reporting is unbiased.
    """
    pnl_curve = _sweep_pnl(predictions, pnl, theta_grid)
    kernel    = np.ones(_SMOOTH_W) / _SMOOTH_W
    smoothed  = np.convolve(pnl_curve, kernel, mode="same")
    best = int(np.argmax(smoothed))
    return float(theta_grid[best]), pnl_curve


# ═══════════════════════════════════════════════════════════════════════════════
# Required PDF interface
# ═══════════════════════════════════════════════════════════════════════════════

def optimal_threshold(
    tau: int,
    df_val: pd.DataFrame = None,
    df_test: pd.DataFrame = None,
    predictions_val: np.ndarray = None,
    predictions_test: np.ndarray = None,
    data_path: str = None,
) -> dict:
    """
    [PDF] §3.5 — required interface.

    Always uses client-specific thresholds (one θ* per client at the given τ),
    per PDF §3.5 schema (each row carries its own θ*).

    Robustness rules applied after the val sweep:
      • Unseen-client fallback: if a client has no val trades, the global θ*
        (argmax over all val trades at this τ) is used.
      • Smoothed argmax (via _argmax_theta): narrow spikes in the PnL curve
        are damped before selection, so sparse noisy signals cannot dominate.

    Returns:
        {
            'theta':          Dict[str, float],   # {ClientA: θ*, …}
            'validation_pnl': float,              # sum across all clients at θ*
            'test_pnl':       float,              # sum across all clients at θ*
        }
    """
    if df_val is None or df_test is None:
        df = _build_features(_load_raw(data_path))
        _, df_val, df_test = _split_by_date(df)
    if predictions_val is None or predictions_test is None:
        model = _load_model()
        predictions_val  = model.predict_proba(_build_matrix_tau(df_val,  tau))[:, 1]
        predictions_test = model.predict_proba(_build_matrix_tau(df_test, tau))[:, 1]

    pnl_val  = df_val[f"pnl_{tau}"].values
    pnl_test = df_test[f"pnl_{tau}"].values
    names_val  = df_val["Name"].values
    names_test = df_test["Name"].values

    # Global θ* over all val trades — used as fallback for unseen clients.
    global_theta, _ = _argmax_theta(predictions_val, pnl_val)

    thetas: Dict[str, float] = {}
    val_pnl_total = 0.0
    test_pnl_total = 0.0

    for cid in CLIENT_IDS:
        mask_v = names_val == cid
        mask_t = names_test == cid

        if mask_v.sum() == 0:
            # No val trades for this client — use global θ* as data-driven default.
            thetas[CLIENT_MAP[cid]] = global_theta
            continue

        c_preds_v = predictions_val[mask_v]
        c_pnl_v   = pnl_val[mask_v]

        theta_star, _ = _argmax_theta(c_preds_v, c_pnl_v)
        thetas[CLIENT_MAP[cid]] = theta_star

        # Internalize iff p ≤ θ* (PDF §3.1)
        val_pnl_total  += float(c_pnl_v[c_preds_v <= theta_star].sum())
        test_pnl_total += float(
            pnl_test[mask_t][predictions_test[mask_t] <= theta_star].sum()
        )

    return {
        "theta":          thetas,
        "validation_pnl": float(val_pnl_total),
        "test_pnl":       float(test_pnl_total),
    }


def plot_pnl_vs_theta(
    df_val: pd.DataFrame = None,
    predictions_val_dict: Dict[int, np.ndarray] = None,
    data_path: str = None,
) -> None:
    """
    [PDF] §3.5 — required interface.

    Plots one PnL-vs-θ curve per client per horizon subplot (2×3 grid), with
    each client's optimal θ* marked on its own curve. This is consistent with
    the client-specific CSV output and directly supports the writeup discussion
    of which clients benefit most from externalization (PDF §3.4 item 4).
    """
    if df_val is None or predictions_val_dict is None:
        df = _build_features(_load_raw(data_path))
        _, df_val, _ = _split_by_date(df)
        model = _load_model()
        predictions_val_dict = {
            tau: model.predict_proba(_build_matrix_tau(df_val, tau))[:, 1]
            for tau in HORIZONS
        }

    names_val = df_val["Name"].values

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, tau in zip(axes.flatten(), HORIZONS):
        preds = predictions_val_dict[tau]
        pnl   = df_val[f"pnl_{tau}"].values

        for i, cid in enumerate(CLIENT_IDS):
            mask = names_val == cid
            if mask.sum() == 0:
                continue
            c_preds = preds[mask]
            c_pnl   = pnl[mask]
            theta_star, curve = _argmax_theta(c_preds, c_pnl)
            color = _CLIENT_COLORS[i]

            ax.plot(THETA_GRID, curve, linewidth=1.3, color=color,
                    label=f"Client{cid}  θ*={theta_star:.2f}")
            ax.axvline(theta_star, color=color, linestyle="--", linewidth=0.8, alpha=0.55)
            ax.scatter([theta_star], [float(curve.max())], color=color, s=45, zorder=5)

        ax.set_title(f"τ = {tau}s", fontsize=11)
        ax.set_xlabel("θ", fontsize=9)
        ax.set_ylabel("Val PnL", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", framealpha=0.7)

    fig.suptitle(
        "Task 4 — Per-Client Validation PnL vs Externalization Threshold θ", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Task 4] Plot saved → {PLOT_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# CSV output & master runner
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_csv(rows_by_tau: Dict[int, Dict[str, Tuple[float, float]]],
                  out_path: str = OUTPUT_CSV) -> str:
    """
    Write task4_results.csv per PDF §3.5 schema.
    rows_by_tau[tau][client_display_name] = (θ*, final_pnl)
    Output is client-major (PDF schema): ClientA-5…ClientA-30, ClientB-5…
    """
    rows = []
    for cid in CLIENT_IDS:
        cname = CLIENT_MAP[cid]
        for tau in HORIZONS:
            theta, fp = rows_by_tau[tau][cname]
            rows.append({
                "client":    cname,
                "τ":         tau,
                "θ*":        theta,
                "final_pnl": fp,
            })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, float_format="%.4f", encoding="utf-8")
    print(f"[Task 4] CSV saved → {out_path}")
    return out_path


def run_task4(data_path: str = None) -> Dict[int, dict]:
    """
    Master function for Task 4.

    1. Load data and split by date (matching task3.py).
    2. Load the saved Task 3 model and compute per-τ predictions on val + test.
    3. For each τ, find the per-client θ* on validation, then evaluate test PnL.
    4. Write task4_results.csv and pnl_vs_theta.png.
    """
    print("=" * 70)
    print("TASK 4: OPTIMAL EXTERNALIZATION THRESHOLD")
    print("=" * 70)

    print("[Task 4] Loading data + features …")
    df = _build_features(_load_raw(data_path))
    _, df_val, df_test = _split_by_date(df)
    print(f"[Task 4] Val: {len(df_val)} rows  |  Test: {len(df_test)} rows")

    print("[Task 4] Loading task3_model.pkl …")
    model = _load_model()

    print("[Task 4] Generating per-τ predictions …")
    preds_val  = {tau: model.predict_proba(_build_matrix_tau(df_val,  tau))[:, 1] for tau in HORIZONS}
    preds_test = {tau: model.predict_proba(_build_matrix_tau(df_test, tau))[:, 1] for tau in HORIZONS}

    rows_by_tau: Dict[int, Dict[str, Tuple[float, float]]] = {}
    results_by_tau: Dict[int, dict] = {}

    for tau in HORIZONS:
        res = optimal_threshold(
            tau,
            df_val=df_val, df_test=df_test,
            predictions_val=preds_val[tau],
            predictions_test=preds_test[tau],
        )
        results_by_tau[tau] = res

        # Per-client test PnL at each θ*
        rows_by_tau[tau] = {}
        names_test = df_test["Name"].values
        pnl_test   = df_test[f"pnl_{tau}"].values
        for cid in CLIENT_IDS:
            cname = CLIENT_MAP[cid]
            theta = res["theta"][cname]
            mask = names_test == cid
            client_pnl = float(pnl_test[mask][preds_test[tau][mask] <= theta].sum())
            rows_by_tau[tau][cname] = (theta, client_pnl)

        print(f"  τ={tau:2d}s  val_pnl={res['validation_pnl']:>12.2f}  "
              f"test_pnl={res['test_pnl']:>12.2f}")

    _generate_csv(rows_by_tau)
    plot_pnl_vs_theta(df_val=df_val, predictions_val_dict=preds_val)

    total_val  = sum(r["validation_pnl"] for r in results_by_tau.values())
    total_test = sum(r["test_pnl"]       for r in results_by_tau.values())
    print(f"\n[Task 4] Total Val PnL  (all τ): {total_val:,.2f}")
    print(f"[Task 4] Total Test PnL (all τ): {total_test:,.2f}")
    print("[Task 4] Complete.")
    return results_by_tau


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_task4(path)
