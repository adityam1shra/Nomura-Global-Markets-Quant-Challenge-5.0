"""
Nomura Quant Challenge 5 — Task 3: Adversity Prediction Model
[PDF] §3.2, 15 marks

Self-contained submission file. Run with:
    python task3.py                      # uses ./trade_data.csv next to the script
    python task3.py /path/to/trade.csv   # override CSV path

Produces:
    task3_results.csv      — averaged metrics (train / validation / test)
    task3_model.pkl        — trained HistGBT + isotonic calibrator (stdlib pickle)

Required interface (PDF §3.3):
    predict_adversity(*args, **kwargs) -> float
    compute_metrics(*args, **kwargs)   -> pd.DataFrame
"""
import os
import sys
import pickle
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ────────────────────────────────────────────────────────────────
HORIZONS      = [5, 10, 15, 20, 25, 30]
N_HORIZONS    = len(HORIZONS)
CLIENT_IDS    = ["A", "B", "C", "D", "E", "F"]
RANDOM_SEED   = 42
TRAIN_DAYS    = 25   # days 1-25  (60 %)
VAL_DAYS      = 8    # days 26-33 (20 %)
MODEL_TRAIN_DAYS = 20  # core training days; held-out isotonic on days 21-25
VOL_WINDOW_N  = 20

# Feature names — index order matches the columns of the stacked matrix
# produced by _build_matrix (BASE_COLS first, then tau appended at the end).
FEATURE_NAMES = [
    "client_id",       # 0   label-encoded Name
    "side",            # 1   +1 / -1
    "volume",          # 2   raw volume
    "log_volume",      # 3   log(volume)
    "spread",          # 4   raw spread
    "hour",            # 5   hour of day
    "minute_of_day",   # 6   minutes since midnight
    "day_of_week",     # 7   0=Mon … 4=Fri
    "eta",             # 8   elapsed fraction of trading day
    "rolling_vol",     # 9   realized volatility (N=20, PDF Eq. 10)
    "vol_spread",      # 10  volume × spread (interaction)
    "rolling_adv",     # 11  client's rolling adverse-trade rate (lag-1, N=50)
    "order_flow",      # 12  cumulative signed volume (reset daily)
    "spread_zscore",   # 13  spread z-score vs client's recent 50-trade window
    "mom_short",       # 14  short-term mid-price drift over last 5 trades
    "tau",             # 15  horizon (stacked-model feature, appended last)
]

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA   = os.path.join(SCRIPT_DIR, "trade_data.csv")
MODEL_PATH     = os.path.join(SCRIPT_DIR, "task3_model.pkl")
OUTPUT_CSV     = os.path.join(SCRIPT_DIR, "task3_results.csv")

T_OPEN_SEC     = 9 * 3600 + 30 * 60   # 09:30
TRADING_SECS   = (16 * 60 - (9 * 60 + 30)) * 60  # 23400 s


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading & feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

def _load_raw(path: str = None) -> pd.DataFrame:
    """Load CSV, parse dates, sort, assign sequential day numbers."""
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
    """
    PDF Eq. 10: sigma_t = sqrt(mean(r_k^2)) over N=20 trades preceding trade t.
    Cross-day returns are zeroed to suppress overnight gaps.
    """
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
    """Compute all feature columns in-place."""
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

    df["rolling_vol"]  = _rolling_volatility(df)
    df["vol_spread"]   = df["volume"] * df["spread"]

    # Rolling adversity rate per client, lag-1 to avoid leakage
    df["adv5_label"] = ((df["Side"] * (df["M5"] - df["Trade Price"])) < 0).astype(float)
    df["rolling_adv"] = (
        df.groupby("Name")["adv5_label"]
          .transform(lambda s: s.shift(1).rolling(50, min_periods=1).mean())
    ).fillna(0.45)

    # Cumulative signed order flow per client, reset each day, lag-1
    df["signed_vol"] = df["Side"] * df["Volume"]
    df["order_flow"] = (
        df.groupby(["Name", "day_num"])["signed_vol"]
          .transform(lambda s: s.cumsum() - s)
    ).astype(float)

    # Spread z-score vs client's recent 50-trade window (lag-1)
    df["spread_zscore"] = (
        df.groupby("Name")["spread"]
          .transform(lambda s: (s - s.shift(1).rolling(50, min_periods=5).mean())
                               / (s.shift(1).rolling(50, min_periods=5).std() + 1e-8))
    ).fillna(0.0)

    # Short-term mid-price momentum over last 5 same-day trades
    df["m0_lag5"] = (
        df.groupby(["Name", "day_num"])["M0"]
          .transform(lambda s: s.shift(5))
    )
    df["mom_short"] = ((df["M0"] - df["m0_lag5"]) / (df["m0_lag5"] + 1e-8)).fillna(0.0)

    # Adversity labels for all horizons
    for tau in HORIZONS:
        df[f"adverse_{tau}"] = ((df["Side"] * (df[f"M{tau}"] - df["Trade Price"])) < 0).astype(int)

    return df


def _split_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """60/20/20 chronological split by date (PDF §3.2)."""
    train = df[df["day_num"] <= TRAIN_DAYS].copy()
    val   = df[(df["day_num"] > TRAIN_DAYS) & (df["day_num"] <= TRAIN_DAYS + VAL_DAYS)].copy()
    test  = df[df["day_num"] > TRAIN_DAYS + VAL_DAYS].copy()
    return train, val, test


BASE_COLS = [
    "client_id", "side", "volume", "log_volume", "spread",
    "hour", "minute_of_day", "day_of_week", "eta", "rolling_vol",
    "vol_spread", "rolling_adv", "order_flow", "spread_zscore", "mom_short",
]


def _build_matrix(df: pd.DataFrame, tau: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build stacked feature matrix (tau as feature) or single-tau matrix.
    Returns (X, y).
    """
    if tau is not None:
        X = np.column_stack([df[BASE_COLS].values, np.full(len(df), float(tau))])
        y = df[f"adverse_{tau}"].values
        return X.astype(np.float64), y.astype(int)

    X_parts, y_parts = [], []
    for t in HORIZONS:
        tau_col = np.full((len(df), 1), float(t))
        X_parts.append(np.column_stack([df[BASE_COLS].values, tau_col]))
        y_parts.append(df[f"adverse_{t}"].values)
    return np.vstack(X_parts).astype(np.float64), np.concatenate(y_parts).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# Required PDF interface
# ═══════════════════════════════════════════════════════════════════════════════

def predict_adversity(features: np.ndarray) -> float:
    """
    [PDF] §3.3 — required interface (arguments left open per PDF).

    Parameters
    ----------
    features : np.ndarray
        1-D array of length 16 matching FEATURE_NAMES order (tau at index 15).

    Returns
    -------
    float
        Calibrated P(adverse at tau) in [0, 1].
    """
    model = _load_model()
    X = np.asarray(features, dtype=np.float64).reshape(1, -1)
    return float(model.predict_proba(X)[0, 1])


def compute_metrics(
    model=None,
    X_train: np.ndarray = None, y_train: np.ndarray = None,
    X_val:   np.ndarray = None, y_val:   np.ndarray = None,
    X_test:  np.ndarray = None, y_test:  np.ndarray = None,
    data_path: str = None,
) -> pd.DataFrame:
    """
    [PDF] §3.3 — required interface (arguments left open per PDF).

    Callable two ways:
      1. compute_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test)
         — uses the matrices you pass in.
      2. compute_metrics()  — loads pkl, loads trade_data.csv, builds matrices,
         then computes metrics. Grader-friendly form.

    Returns DataFrame with rows=[train, validation, test] and
    columns=[accuracy, precision, recall, log_loss], averaged across τ.
    """
    if model is None:
        model = _load_model()
    if X_train is None:
        df = _load_raw(data_path)
        df = _build_features(df)
        df_tr, df_va, df_te = _split_by_date(df)
        X_train, y_train = _build_matrix(df_tr)
        X_val,   y_val   = _build_matrix(df_va)
        X_test,  y_test  = _build_matrix(df_te)

    rows = {}
    for name, X, y in [
        ("train",      X_train, y_train),
        ("validation", X_val,   y_val),
        ("test",       X_test,  y_test),
    ]:
        rows[name] = _metrics_averaged(model, X, y)
    return pd.DataFrame(rows).T[["accuracy", "precision", "recall", "log_loss"]]


def _metrics_averaged(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute metrics per horizon then return arithmetic mean."""
    n = len(y)
    assert n % N_HORIZONS == 0
    n_per = n // N_HORIZONS
    acc, prec, rec, ll = [], [], [], []
    for i in range(N_HORIZONS):
        Xt = X[i * n_per: (i + 1) * n_per]
        yt = y[i * n_per: (i + 1) * n_per]
        prob = model.predict_proba(Xt)[:, 1]
        pred = (prob >= 0.5).astype(int)
        acc.append(accuracy_score(yt, pred))
        prec.append(precision_score(yt, pred, zero_division=0.0))
        rec.append(recall_score(yt, pred, zero_division=0.0))
        ll.append(log_loss(yt, prob, labels=[0, 1]))
    return {
        "accuracy":  float(np.mean(acc)),
        "precision": float(np.mean(prec)),
        "recall":    float(np.mean(rec)),
        "log_loss":  float(np.mean(ll)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class _IsotonicCalibrator:
    """Wraps a fitted HistGBT and applies isotonic post-hoc calibration."""

    def __init__(self, base):
        self.base = base
        self.iso  = IsotonicRegression(out_of_bounds="clip")

    def fit(self, X: np.ndarray, y: np.ndarray):
        raw = self.base.predict_proba(X)[:, 1]
        self.iso.fit(raw, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.iso.transform(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_and_save(data_path: str = None):
    """
    Train HistGBT with held-out isotonic calibration and persist the artefact.

    Pipeline:
      1. Load + engineer features.
      2. Chronological 60/20/20 split.
      3. Sub-split train (days 1-25) → core (1-20) and calibration (21-25).
      4. HistGradientBoostingClassifier on days 1-20 with built-in early stopping.
      5. IsotonicRegression on the held-out days 21-25.
      6. Pickle the components.
    """
    print("[Task 3] Loading and engineering features …")
    df = _load_raw(data_path)
    df = _build_features(df)

    df_train, df_val, df_test = _split_by_date(df)
    df_model_train = df_train[df_train["day_num"] <= MODEL_TRAIN_DAYS].copy()
    df_cal         = df_train[df_train["day_num"] >  MODEL_TRAIN_DAYS].copy()

    print(f"[Task 3] Splits — model_train: {len(df_model_train)} | "
          f"cal: {len(df_cal)} | val: {len(df_val)} | test: {len(df_test)}")

    X_mt, y_mt   = _build_matrix(df_model_train)
    X_cal, y_cal = _build_matrix(df_cal)
    X_val, y_val = _build_matrix(df_val)
    X_test, y_test = _build_matrix(df_test)
    X_train, y_train = _build_matrix(df_train)

    print(f"[Task 3] X_model_train: {X_mt.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
    print(f"[Task 3] Base adverse rate — train: {y_mt.mean():.4f}  val: {y_val.mean():.4f}  test: {y_test.mean():.4f}")

    hgbt = HistGradientBoostingClassifier(
        max_iter=800,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        random_state=RANDOM_SEED,
    )
    hgbt.fit(X_mt, y_mt)
    print(f"[Task 3] HistGBT stopped at iteration {hgbt.n_iter_}")

    model = _IsotonicCalibrator(hgbt)
    model.fit(X_cal, y_cal)
    print("[Task 3] Isotonic calibration done (on held-out days 21-25).")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"hgbt": hgbt, "iso": model.iso}, f)
    print(f"[Task 3] Model saved → {MODEL_PATH}")

    return model, X_train, y_train, X_val, y_val, X_test, y_test


def _load_model() -> _IsotonicCalibrator:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run task3.py first to train."
        )
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    # Reconstruct from stored sklearn components — avoids __main__ binding issues.
    cal = _IsotonicCalibrator(data["hgbt"])
    cal.iso = data["iso"]
    return cal


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers for Tasks 4 & 5
# ═══════════════════════════════════════════════════════════════════════════════

def get_predictions_for_task4(
    model, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> Dict[str, Dict[int, np.ndarray]]:
    """Return per-horizon P(adverse) arrays for val and test splits."""
    result: Dict[str, Dict[int, np.ndarray]] = {"val": {}, "test": {}}
    for split_name, df_split in [("val", df_val), ("test", df_test)]:
        for tau in HORIZONS:
            X_tau, _ = _build_matrix(df_split, tau)
            result[split_name][tau] = model.predict_proba(X_tau)[:, 1]
    return result


def get_alpha_batch(base_features: np.ndarray, model=None) -> np.ndarray:
    """
    Mean P(adverse) across all 6 horizons for each row.
    base_features: (n, 15) — FEATURE_NAMES without the trailing tau column.
    """
    if model is None:
        model = _load_model()
    base = np.asarray(base_features, dtype=np.float64)
    if base.ndim == 1:
        base = base.reshape(1, -1)
    n = base.shape[0]
    probs = np.zeros((n, N_HORIZONS))
    for idx, tau in enumerate(HORIZONS):
        tau_col = np.full((n, 1), float(tau))
        X = np.hstack([base, tau_col])
        probs[:, idx] = model.predict_proba(X)[:, 1]
    return probs.mean(axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV output & master runner
# ═══════════════════════════════════════════════════════════════════════════════

def generate_task3_csv(metrics_df: pd.DataFrame, out_path: str = OUTPUT_CSV) -> str:
    """Write task3_results.csv per PDF §3.3 schema."""
    out = metrics_df[["accuracy", "precision", "recall", "log_loss"]].copy()
    out.index.name = "split"
    out.to_csv(out_path, float_format="%.6f", encoding="utf-8")
    print(f"[Task 3] CSV saved → {out_path}")
    return out_path


def evaluate_and_report(data_path: str = None) -> pd.DataFrame:
    """
    Load the persisted ensemble, build matrices, compute + write metrics.
    Independently invocable: assumes task3_model.pkl already exists.
    """
    print("=" * 70)
    print("TASK 3: EVALUATING SAVED MODEL")
    print("=" * 70)
    metrics = compute_metrics(data_path=data_path)
    print("\n── Averaged metrics (across 6 horizons) ──")
    print(metrics.to_string(float_format="%.4f"))
    generate_task3_csv(metrics)
    return metrics


def run_task3(data_path: str = None) -> pd.DataFrame:
    """Train, save, evaluate, and write the results CSV."""
    print("=" * 70)
    print("TASK 3: ADVERSITY PREDICTION MODEL")
    print("=" * 70)
    model, X_train, y_train, X_val, y_val, X_test, y_test = train_and_save(data_path)
    metrics = compute_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test)
    print("\n── Averaged metrics (across 6 horizons) ──")
    print(metrics.to_string(float_format="%.4f"))
    generate_task3_csv(metrics)
    print("\n[Task 3] Complete.")
    return metrics


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_task3(path)
