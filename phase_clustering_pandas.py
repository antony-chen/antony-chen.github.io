"""
phase_clustering_pandas.py
--------------------------
Cluster feeder voltage profiles into phases A, B, C from a pandas DataFrame.

Expected input DataFrame columns:
    PHASE_CHILD  str   - noisy phase label ('A', 'B', 'C', or NaN/None)
    ENDPOINTID   any   - device identifier
    TIMESTAMP    any   - 15-minute interval timestamp (sortable)
    DATA         float - voltage reading (volts)

Output: input DataFrame with three new columns merged on ENDPOINTID:
    PREDICTED_PHASE    str   - predicted phase letter
    PHASE_CONFIDENCE   float - model confidence in [0, 1]
    IS_SUSPECT_LABEL   bool  - True when PHASE_CHILD contradicts the
                               prediction at high confidence (likely mislabeled)

Algorithm overview
------------------
1. Pivot the long-format DataFrame into a wide voltage matrix
   (n_devices × n_time_slots) and linearly interpolate any gaps.

2. Spectral clustering on the pairwise Pearson-correlation affinity matrix.
   Devices on the same phase share load events and voltage-drop patterns,
   producing high within-phase correlation regardless of absolute voltage
   level.  This step is entirely unsupervised and immune to label noise.

3. Gaussian Mixture Model on PCA-compressed daily-profile features for
   soft per-device probabilities used in confidence scoring and mislabel
   detection.

4. Confidence-weighted majority vote maps anonymous cluster integers to
   phase letters A/B/C using the (noisy) training labels.

5. Devices whose stored label disagrees with the high-confidence prediction
   are flagged as IS_SUSPECT_LABEL for downstream review.

Dependencies: numpy, pandas, scikit-learn
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PHASE_TO_INT: dict[str, int] = {"A": 0, "B": 1, "C": 2}
_INT_TO_PHASE: dict[int, str] = {0: "A", 1: "B", 2: "C"}

# Above this device count the N×N correlation matrix becomes expensive;
# fall back to GMM-only clustering.
_SPECTRAL_DEVICE_LIMIT = 10_000

# 15-minute intervals → 96 slots per 24-hour day
_SLOTS_PER_DAY = 96


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_phase_profiles(
    df: pd.DataFrame,
    *,
    phase_col: str = "PHASE_CHILD",
    device_col: str = "ENDPOINTID",
    time_col: str = "TIMESTAMP",
    voltage_col: str = "DATA",
    n_phases: int = 3,
    mislabel_threshold: float = 0.80,
    use_spectral: bool = True,
) -> pd.DataFrame:
    """
    Cluster feeder voltage profiles into phases A, B, C.

    Parameters
    ----------
    df                  Pandas DataFrame (see module docstring for schema).
    phase_col           Column containing the (noisy) phase label.
    device_col          Column containing the device identifier.
    time_col            Column containing 15-minute interval timestamps.
    voltage_col         Column containing voltage readings in volts.
    n_phases            Number of phases (almost always 3).
    mislabel_threshold  Confidence above which a disagreeing label is flagged.
    use_spectral        Use spectral clustering (recommended) when device count
                        is within _SPECTRAL_DEVICE_LIMIT.

    Returns
    -------
    Input DataFrame joined with PREDICTED_PHASE, PHASE_CONFIDENCE,
    IS_SUSPECT_LABEL columns keyed on device_col.
    """

    # ------------------------------------------------------------------
    # Step 1  Assign a 0-based integer time-slot index from the sorted
    #         distinct timestamps so every device maps to the same columns.
    # ------------------------------------------------------------------
    sorted_times = sorted(df[time_col].unique())
    time_to_slot = {t: i for i, t in enumerate(sorted_times)}
    n_slots = len(sorted_times)
    _log(f"{n_slots} time slots found ({n_slots / _SLOTS_PER_DAY:.1f} days of 15-min data)")

    # ------------------------------------------------------------------
    # Step 2  Collapse duplicate (device, timestamp) readings by averaging,
    #         then pivot to wide format: one row per device, one col per slot.
    # ------------------------------------------------------------------
    df_work = df[[device_col, phase_col, time_col, voltage_col]].copy()
    df_work["_slot"] = df_work[time_col].map(time_to_slot)

    # Keep one phase label per device (majority label, ignoring NaN)
    device_labels = (
        df_work.dropna(subset=[phase_col])
        .groupby(device_col)[phase_col]
        .agg(lambda s: s.mode().iloc[0] if not s.empty else np.nan)
    )

    # Average duplicate readings, then pivot
    agg = (
        df_work
        .groupby([device_col, "_slot"], sort=False)[voltage_col]
        .mean()
        .reset_index()
    )
    V_wide = agg.pivot(index=device_col, columns="_slot", values=voltage_col)
    V_wide = V_wide.reindex(columns=range(n_slots))   # ensure all slots present

    n_devices = len(V_wide)
    _log(f"Voltage matrix: {n_devices:,} devices × {n_slots} slots")

    # ------------------------------------------------------------------
    # Step 3  Extract arrays.
    # ------------------------------------------------------------------
    device_ids = V_wide.index.to_numpy()
    V          = V_wide.to_numpy(dtype=float)          # (n_devices, n_slots)

    raw_labels = device_labels.reindex(device_ids).to_numpy()

    # ------------------------------------------------------------------
    # Step 4  Impute missing slots per device via linear interpolation.
    # ------------------------------------------------------------------
    V = _impute(V)

    # ------------------------------------------------------------------
    # Step 5  Encode noisy labels: A→0, B→1, C→2, anything else → -1.
    # ------------------------------------------------------------------
    noisy = np.array([
        _PHASE_TO_INT.get(str(lbl).strip().upper(), -1)
        if pd.notna(lbl) else -1
        for lbl in raw_labels
    ])
    n_labeled = int((noisy >= 0).sum())
    _log(f"Labeled devices: {n_labeled:,}  |  Unlabeled: {len(noisy) - n_labeled:,}")

    # ------------------------------------------------------------------
    # Step 6  Primary clustering — spectral when feasible, else GMM.
    #
    #  Spectral clustering uses the N×N Pearson correlation matrix as an
    #  affinity graph.  Devices on the same phase experience correlated
    #  load events and produce highly similar voltage trajectories, making
    #  correlation a strong phase-membership signal independent of labels.
    # ------------------------------------------------------------------
    if use_spectral and n_devices <= _SPECTRAL_DEVICE_LIMIT:
        _log("Running spectral clustering on pairwise correlation affinity …")
        hard_ids = _spectral_cluster(V, n_phases)
    else:
        if use_spectral:
            _log(
                f"Device count {n_devices:,} > limit {_SPECTRAL_DEVICE_LIMIT:,}"
                " — skipping spectral, using GMM hard labels"
            )
        hard_ids = None

    # ------------------------------------------------------------------
    # Step 7  GMM on PCA-compressed daily-profile features.
    #         Always runs; provides the soft probability matrix used for
    #         confidence scoring and mislabel detection in Steps 8-9.
    # ------------------------------------------------------------------
    _log("Running GMM for soft probability estimates …")
    probs, gmm_ids = _gmm_cluster(V, n_phases, n_slots)

    # Align GMM cluster integers to match the spectral cluster integers
    # so that probs[:, c] corresponds to the same group as hard_ids == c.
    if hard_ids is not None:
        mapping = _majority_mapping(gmm_ids, hard_ids, n_phases)
        gmm_ids = np.array([mapping[g] for g in gmm_ids])   # noqa: F841
        probs   = _rearrange_cols(probs, mapping)
    else:
        hard_ids = gmm_ids

    # ------------------------------------------------------------------
    # Step 8  Map anonymous cluster integers → phase letters.
    #         Use a confidence-weighted majority vote over labeled devices
    #         so high-confidence predictions drown out noisy labels.
    # ------------------------------------------------------------------
    cluster_to_phase = _assign_phases(hard_ids, noisy, probs, n_phases)
    _log(f"Cluster → Phase assignment: {cluster_to_phase}")

    predicted  = np.array([cluster_to_phase[c] for c in hard_ids])
    confidence = probs.max(axis=1)

    # ------------------------------------------------------------------
    # Step 9  Flag suspect labels.
    #         A labeled device is suspect when the model is confident in
    #         a different phase than the stored label.
    # ------------------------------------------------------------------
    pred_int = np.vectorize(_PHASE_TO_INT.get)(predicted)
    suspect  = (noisy >= 0) & (pred_int != noisy) & (confidence > mislabel_threshold)
    _log(f"Suspect labels: {int(suspect.sum()):,} / {n_labeled:,} labeled devices")

    # ------------------------------------------------------------------
    # Step 10  Build result DataFrame and merge back onto the original.
    # ------------------------------------------------------------------
    result = pd.DataFrame({
        device_col:         device_ids,
        "PREDICTED_PHASE":  predicted,
        "PHASE_CONFIDENCE": np.round(confidence, 4).astype(float),
        "IS_SUSPECT_LABEL": suspect.astype(bool),
    })

    return df.merge(result, on=device_col, how="left")


def validate_clusters(df_result: pd.DataFrame, device_col: str = "ENDPOINTID") -> None:
    """
    Print per-phase device count and mean confidence as a sanity check.
    A good clustering shows mean confidence > 0.80 across all phases.
    """
    summary = (
        df_result
        .drop_duplicates(subset=[device_col])
        [["PREDICTED_PHASE", "PHASE_CONFIDENCE"]]
        .groupby("PREDICTED_PHASE")
        .agg(count=("PHASE_CONFIDENCE", "count"),
             mean_confidence=("PHASE_CONFIDENCE", "mean"))
    )
    print("\n── Cluster validation ──────────────────────────────")
    print(summary.to_string())
    print("────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[phase_cluster] {msg}")


def _impute(V: np.ndarray) -> np.ndarray:
    """
    Per-device linear interpolation along the time axis.
    Leading/trailing NaN gaps are filled by propagating the nearest valid
    reading.  All-NaN rows (devices with no data) are filled with the
    global mean voltage.
    """
    out = V.copy()
    global_mean = float(np.nanmean(V)) if not np.isnan(V).all() else 0.0

    for i in range(out.shape[0]):
        s = pd.Series(out[i])
        filled = s.interpolate(method="linear", limit_direction="both")
        out[i] = filled.to_numpy()
        if np.isnan(out[i]).any():   # entire row was NaN
            out[i] = global_mean

    return out


def _spectral_cluster(V: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Build a Pearson-correlation affinity matrix and apply spectral clustering.

    Each device's voltage series is z-scored before computing correlations so
    that absolute voltage offsets do not dominate — only load-profile shape
    matters.
    """
    V_z      = StandardScaler().fit_transform(V.T).T    # z-score row-wise
    corr     = np.corrcoef(V_z)                         # (n, n)
    corr     = np.nan_to_num(corr, nan=0.0)
    affinity = np.clip((corr + 1.0) / 2.0, 0.0, 1.0)   # map [-1, 1] → [0, 1]

    return SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        n_init=10,
        random_state=42,
    ).fit_predict(affinity)


def _gmm_cluster(
    V: np.ndarray,
    n_clusters: int,
    n_slots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster using a Gaussian Mixture Model on compressed daily-profile features.

    Features
    --------
    - Mean voltage at each 15-min slot averaged over all full days
      (captures the daily load shape distinguishing customers on each phase).
    - Standard deviation at each slot across days (day-to-day spread).
    - Global mean, std, 5th and 95th percentile (overall voltage level/range).

    PCA is applied before GMM to decorrelate features and speed up fitting.

    Returns (soft_probs, hard_labels).
    """
    n_full_days = n_slots // _SLOTS_PER_DAY
    V_trim      = V[:, : n_full_days * _SLOTS_PER_DAY]

    by_day     = V_trim.reshape(-1, n_full_days, _SLOTS_PER_DAY)
    daily_mean = by_day.mean(axis=1)    # (n, 96)
    daily_std  = by_day.std(axis=1)     # (n, 96)

    features = np.column_stack([
        daily_mean,
        daily_std,
        V.mean(axis=1),
        V.std(axis=1),
        np.percentile(V,  5, axis=1),
        np.percentile(V, 95, axis=1),
    ])

    X     = StandardScaler().fit_transform(features)
    n_pca = min(20, X.shape[1], X.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        n_init=20,
        random_state=42,
    ).fit(X_pca)

    probs = gmm.predict_proba(X_pca)
    ids   = gmm.predict(X_pca)
    return probs, ids


def _majority_mapping(
    source: np.ndarray,
    target: np.ndarray,
    n: int,
) -> dict[int, int]:
    """Map each source cluster to the most common target cluster among its members."""
    mapping: dict[int, int] = {}
    for s in range(n):
        mask = source == s
        mapping[s] = int(np.bincount(target[mask], minlength=n).argmax()) if mask.any() else s
    return mapping


def _rearrange_cols(probs: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Reorder probability columns so probs[:, new] holds what was in probs[:, orig]."""
    out = np.zeros_like(probs)
    for orig, new in mapping.items():
        out[:, new] = probs[:, orig]
    return out


def _assign_phases(
    cluster_ids: np.ndarray,
    noisy: np.ndarray,
    probs: np.ndarray,
    n_clusters: int,
) -> dict[int, str]:
    """
    Map anonymous cluster integers → phase letters via confidence-weighted
    majority vote over labeled devices.

    Weighting by model confidence means edge-case devices (near cluster
    boundaries, more likely to be mislabeled) contribute less to the vote.
    Clusters with no labeled members receive the leftover phase letter.
    """
    assigned: dict[int, str] = {}
    remaining = list(_INT_TO_PHASE.values())   # ['A', 'B', 'C']
    labeled   = noisy >= 0

    for c in range(n_clusters):
        mask = (cluster_ids == c) & labeled
        if not mask.any():
            continue

        weights    = probs[mask, c]
        phase_ints = noisy[mask]
        scores: dict[int, float] = {}
        for pi, w in zip(phase_ints, weights):
            scores[pi] = scores.get(pi, 0.0) + float(w)

        best_phase = _INT_TO_PHASE[max(scores, key=scores.get)]
        assigned[c] = best_phase
        if best_phase in remaining:
            remaining.remove(best_phase)

    for c in range(n_clusters):
        if c not in assigned:
            assigned[c] = remaining.pop(0) if remaining else "?"

    return assigned


# ---------------------------------------------------------------------------
# Quick-start example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime, random

    rng = np.random.default_rng(0)
    random.seed(0)

    # ---- Synthetic dataset -------------------------------------------------
    # 3 phases × 50 devices, one week of 15-min readings (672 slots)
    n_per_phase = 50
    timestamps  = [
        datetime.datetime(2024, 1, 1) + datetime.timedelta(minutes=15 * i)
        for i in range(7 * _SLOTS_PER_DAY)
    ]

    def phase_voltage(slot: int, phase: int, device_noise: float) -> float:
        base  = 120.0 + phase * 2.0
        daily = 5.0 * np.sin(2 * np.pi * (slot % _SLOTS_PER_DAY) / _SLOTS_PER_DAY + phase * 0.8)
        noise = rng.normal(0, 0.5) + device_noise
        return round(base + daily + noise, 4)

    rows: list[dict] = []
    ground_truth: list[dict] = []

    for phase_idx, letter in enumerate(("A", "B", "C")):
        for dev in range(n_per_phase):
            dev_id    = f"DEV_{letter}_{dev:03d}"
            dev_noise = rng.normal(0, 0.3)
            # Inject ~5 % label errors
            label = letter if random.random() > 0.05 else random.choice(["A", "B", "C"])
            for slot, ts in enumerate(timestamps):
                rows.append({
                    "PHASE_CHILD": label,
                    "ENDPOINTID":  dev_id,
                    "TIMESTAMP":   ts,
                    "DATA":        phase_voltage(slot, phase_idx, dev_noise),
                })
            ground_truth.append({"ENDPOINTID": dev_id, "TRUE_PHASE": letter})

    df_input = pd.DataFrame(rows)
    df_truth = pd.DataFrame(ground_truth)

    print(f"Input shape: {df_input.shape}")
    print(df_input.head())

    # ---- Run clustering ----------------------------------------------------
    df_result = cluster_phase_profiles(df_input)
    validate_clusters(df_result)

    # ---- Evaluate against ground truth -------------------------------------
    eval_df = (
        df_result
        .drop_duplicates(subset=["ENDPOINTID"])
        [["ENDPOINTID", "PREDICTED_PHASE", "IS_SUSPECT_LABEL"]]
        .merge(df_truth, on="ENDPOINTID")
    )
    correct = (eval_df["PREDICTED_PHASE"] == eval_df["TRUE_PHASE"]).sum()
    total   = len(eval_df)
    print(f"Accuracy on synthetic data: {correct}/{total} ({100 * correct / total:.1f} %)")

    print("\nSuspect-labeled devices (sample):")
    print(eval_df[eval_df["IS_SUSPECT_LABEL"]].head(10).to_string(index=False))
