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

Algorithm
---------
1.  Pivot long-format data into a wide voltage matrix (n_devices × n_slots).

2.  Build an N×N affinity matrix from a 70/30 blend of:
      - First-difference (dV) Pearson correlation  — removes the shared diurnal
        common-mode signal and amplifies load-switching events, which are the
        strongest phase-membership signal at 15-min resolution.
      - Raw voltage Pearson correlation             — adds stability on lightly
        loaded feeders where step-change signal is weak.
    Imputed gap segments are masked before differencing so artificially smooth
    interpolated runs do not inflate similarity.

3.  Spectral clustering on the affinity matrix → hard phase assignments.

4.  Soft probabilities derived directly from the affinity matrix:
      prob(device i → cluster c) ∝ mean affinity from i to all members of c.
    This replaces the previous GMM step entirely. GMM was collapsing 90 % of
    devices into one cluster because its EM algorithm converges to a degenerate
    solution when features are not well-separated. Affinity-based probabilities
    have no optimisation step and therefore no collapse risk.

5.  Confidence-weighted majority vote maps cluster integers → A/B/C using the
    (noisy) training labels.

6.  Devices whose label disagrees with a high-confidence prediction are flagged
    as IS_SUSPECT_LABEL.

Large-fleet fallback (> 10 000 devices)
----------------------------------------
The N×N correlation matrix becomes expensive above ~10 k devices. The fallback
uses K-Means on fleet-residual daily-profile features (daily mean minus
fleet-wide median per slot) which removes the common-mode signal before
clustering. Soft probabilities come from inverse distance to centroids.

Dependencies: numpy, pandas, scikit-learn
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PHASE_TO_INT: dict[str, int] = {"A": 0, "B": 1, "C": 2}
_INT_TO_PHASE: dict[int, str] = {0: "A", 1: "B", 2: "C"}
_SPECTRAL_DEVICE_LIMIT = 10_000   # N×N matrix threshold
_SLOTS_PER_DAY = 96               # 15-min intervals × 24 h


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

    Returns
    -------
    Input DataFrame joined with PREDICTED_PHASE, PHASE_CONFIDENCE,
    IS_SUSPECT_LABEL columns keyed on device_col.
    """

    # ------------------------------------------------------------------
    # Step 1  Time-slot index from sorted distinct timestamps.
    # ------------------------------------------------------------------
    sorted_times = sorted(df[time_col].unique())
    n_slots      = len(sorted_times)
    time_to_slot = {t: i for i, t in enumerate(sorted_times)}
    _log(f"{n_slots} time slots ({n_slots / _SLOTS_PER_DAY:.1f} days of 15-min data)")

    # ------------------------------------------------------------------
    # Step 2  Collapse duplicate (device, slot) readings by averaging,
    #         keep one phase label per device (majority vote), pivot.
    # ------------------------------------------------------------------
    df_work = df[[device_col, phase_col, time_col, voltage_col]].copy()
    df_work["_slot"] = df_work[time_col].map(time_to_slot)

    device_labels = (
        df_work.dropna(subset=[phase_col])
        .groupby(device_col)[phase_col]
        .agg(lambda s: s.mode().iloc[0] if not s.empty else np.nan)
    )

    agg    = df_work.groupby([device_col, "_slot"], sort=False)[voltage_col].mean().reset_index()
    V_wide = agg.pivot(index=device_col, columns="_slot", values=voltage_col).reindex(columns=range(n_slots))

    n_devices  = len(V_wide)
    device_ids = V_wide.index.to_numpy()
    V          = V_wide.to_numpy(dtype=float)
    raw_labels = device_labels.reindex(device_ids).to_numpy()
    _log(f"Voltage matrix: {n_devices:,} devices × {n_slots} slots")

    # ------------------------------------------------------------------
    # Step 3  Impute missing slots (linear interpolation per device).
    #         gap_mask records which values were originally missing so
    #         imputed segments can be excluded from differencing later.
    # ------------------------------------------------------------------
    V, gap_mask = _impute(V)

    # ------------------------------------------------------------------
    # Step 4  Encode noisy labels: A→0, B→1, C→2, unknown→-1.
    # ------------------------------------------------------------------
    noisy = np.array([
        _PHASE_TO_INT.get(str(lbl).strip().upper(), -1)
        if pd.notna(lbl) else -1
        for lbl in raw_labels
    ])
    n_labeled = int((noisy >= 0).sum())
    _log(f"Labeled: {n_labeled:,}  |  Unlabeled: {len(noisy) - n_labeled:,}")

    # ------------------------------------------------------------------
    # Step 5  Cluster.
    # ------------------------------------------------------------------
    if n_devices <= _SPECTRAL_DEVICE_LIMIT:
        _log("Building first-difference correlation affinity matrix …")
        affinity = _build_affinity(V, gap_mask)

        _log("Running spectral clustering …")
        hard_ids = _spectral_cluster(affinity, n_phases)

        _log("Computing affinity-based soft probabilities …")
        probs = _affinity_soft_probs(affinity, hard_ids, n_phases)
    else:
        _log(f"{n_devices:,} devices > {_SPECTRAL_DEVICE_LIMIT:,} limit — using K-Means fallback …")
        hard_ids, probs = _kmeans_fallback(V, n_phases, n_slots)

    # ------------------------------------------------------------------
    # Step 6  Map cluster integers → phase letters via confidence-weighted
    #         majority vote over labeled devices.
    # ------------------------------------------------------------------
    cluster_to_phase = _assign_phases(hard_ids, noisy, probs, n_phases)
    _log(f"Cluster → Phase: {cluster_to_phase}")

    predicted  = np.array([cluster_to_phase[c] for c in hard_ids])
    confidence = probs.max(axis=1)

    # ------------------------------------------------------------------
    # Step 7  Flag suspect labels.
    # ------------------------------------------------------------------
    pred_int = np.vectorize(_PHASE_TO_INT.get)(predicted)
    suspect  = (noisy >= 0) & (pred_int != noisy) & (confidence > mislabel_threshold)
    _log(f"Suspect labels: {int(suspect.sum()):,} / {n_labeled:,}")

    # ------------------------------------------------------------------
    # Step 8  Merge results back onto the original DataFrame.
    # ------------------------------------------------------------------
    result = pd.DataFrame({
        device_col:         device_ids,
        "PREDICTED_PHASE":  predicted,
        "PHASE_CONFIDENCE": np.round(confidence, 4).astype(float),
        "IS_SUSPECT_LABEL": suspect.astype(bool),
    })
    return df.merge(result, on=device_col, how="left")


def validate_clusters(df_result: pd.DataFrame, device_col: str = "ENDPOINTID") -> None:
    """Print per-phase device count and mean confidence as a sanity check."""
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
# Core helpers
# ---------------------------------------------------------------------------

def _impute(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear interpolation per device; fill leading/trailing gaps with nearest
    valid value. All-NaN devices receive the global mean voltage.

    Returns (V_filled, gap_mask) where gap_mask is True where values were
    originally missing.
    """
    out      = V.copy()
    gap_mask = np.isnan(V)
    global_mean = float(np.nanmean(V)) if not np.isnan(V).all() else 0.0

    for i in range(out.shape[0]):
        out[i] = (
            pd.Series(out[i])
            .interpolate(method="linear", limit_direction="both")
            .to_numpy()
        )
        if np.isnan(out[i]).any():
            out[i] = global_mean

    return out, gap_mask


def _build_affinity(V: np.ndarray, gap_mask: np.ndarray) -> np.ndarray:
    """
    Build the N×N affinity matrix used by spectral clustering.

    70 % first-difference correlation + 30 % raw-voltage correlation.

    First-difference correlation
    ----------------------------
    dV[t] = V[t] - V[t-1] acts as a high-pass filter that cancels the
    large fleet-wide diurnal common-mode signal and leaves load-switching
    events — the strongest per-phase signal at 15-min resolution.
    Gap boundaries are masked to NaN before differencing so imputed
    segments don't create artificial correlation.

    Raw-voltage correlation
    -----------------------
    Added at 30 % weight to maintain stability on lightly loaded feeders
    where step-change amplitude is small.
    """
    # ── First-difference correlation ──────────────────────────────────
    V_masked = V.astype(float).copy()
    V_masked[gap_mask] = np.nan
    dV = np.diff(V_masked, axis=1)   # (n, n_slots-1); NaN at gap boundaries

    # Z-score each device's dV series independently (ignore NaN)
    dV_z = np.zeros_like(dV)
    for i in range(dV.shape[0]):
        row   = dV[i]
        valid = ~np.isnan(row)
        if valid.sum() > 1:
            mu  = row[valid].mean()
            sig = row[valid].std()
            dV_z[i] = np.where(valid, (row - mu) / (sig if sig > 0 else 1.0), 0.0)

    corr_dv = np.nan_to_num(np.corrcoef(dV_z), nan=0.0)

    # ── Raw-voltage correlation ───────────────────────────────────────
    V_z     = StandardScaler().fit_transform(V.T).T
    corr_rv = np.nan_to_num(np.corrcoef(V_z), nan=0.0)

    # ── Blend and map to [0, 1] ───────────────────────────────────────
    corr = 0.7 * corr_dv + 0.3 * corr_rv
    return np.clip((corr + 1.0) / 2.0, 0.0, 1.0)


def _spectral_cluster(affinity: np.ndarray, n_clusters: int) -> np.ndarray:
    """Spectral clustering on a precomputed affinity matrix."""
    return SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        n_init=10,
        random_state=42,
    ).fit_predict(affinity)


def _affinity_soft_probs(
    affinity: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Derive soft cluster probabilities directly from the affinity matrix.

    For each device i and cluster c:
        raw[i, c] = mean affinity from device i to all members of cluster c
                    (self is excluded when i is itself in cluster c)

    Rows are then normalised to sum to 1, giving interpretable probabilities.

    This replaces the GMM step. GMM's EM algorithm was collapsing 90 % of
    devices into one cluster because the feature-space distribution was
    near-unimodal. This method has no optimisation loop and therefore no
    collapse risk — the probabilities are a direct read-off of the pairwise
    similarity structure that spectral clustering already computed.
    """
    n = len(labels)

    # Vectorised: affinity @ M gives the sum of affinities to each cluster
    M            = np.zeros((n, n_clusters))
    M[np.arange(n), labels] = 1.0
    cluster_size = M.sum(axis=0)           # (n_clusters,)

    raw = (affinity @ M) / np.maximum(cluster_size, 1)   # (n, n_clusters)

    # Remove self-affinity from each device's own-cluster mean so a device
    # with affinity 1.0 to itself doesn't inflate its own confidence.
    for i in range(n):
        c = labels[i]
        sz = cluster_size[c]
        if sz > 1:
            raw[i, c] = (raw[i, c] * sz - affinity[i, i]) / (sz - 1)

    # Normalise rows to probabilities
    row_sums = raw.sum(axis=1, keepdims=True)
    return raw / np.where(row_sums == 0, 1.0, row_sums)


def _kmeans_fallback(
    V: np.ndarray,
    n_phases: int,
    n_slots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-Means on fleet-residual daily-profile features for large device counts
    where building an N×N affinity matrix is prohibitive.

    Fleet-residual = daily mean profile minus the fleet-wide median profile.
    This removes the common diurnal signal before clustering so the phase-
    specific deviations drive the distance metric.

    Soft probabilities come from inverse centroid distance (softmax).
    """
    n            = V.shape[0]
    n_full_days  = n_slots // _SLOTS_PER_DAY
    V_trim       = V[:, : n_full_days * _SLOTS_PER_DAY]
    daily_mean   = V_trim.reshape(n, n_full_days, _SLOTS_PER_DAY).mean(axis=1)  # (n, 96)
    fleet_median = np.median(daily_mean, axis=0)                                  # (96,)
    residual     = daily_mean - fleet_median                                       # (n, 96)

    # Append first-difference summary stats
    dV       = np.diff(V, axis=1)
    dV_feats = np.column_stack([np.abs(dV).mean(axis=1), dV.std(axis=1)])

    X     = RobustScaler().fit_transform(np.column_stack([residual, dV_feats]))
    n_pca = min(10, X.shape[1], n - 1)
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)

    km       = KMeans(n_clusters=n_phases, n_init=20, random_state=42)
    hard_ids = km.fit_predict(X_pca)

    # Soft probs: softmax of negative normalised centroid distances
    dists    = km.transform(X_pca)                     # (n, n_phases)
    neg_norm = -(dists / dists.mean())
    exp      = np.exp(neg_norm - neg_norm.max(axis=1, keepdims=True))  # numerically stable
    probs    = exp / exp.sum(axis=1, keepdims=True)

    return hard_ids, probs


# ---------------------------------------------------------------------------
# Label-assignment helpers
# ---------------------------------------------------------------------------

def _assign_phases(
    cluster_ids: np.ndarray,
    noisy: np.ndarray,
    probs: np.ndarray,
    n_clusters: int,
) -> dict[int, str]:
    """
    Map cluster integers → phase letters via confidence-weighted majority vote
    over labeled devices. Devices the model is uncertain about (low confidence,
    likely near a boundary or mislabeled) contribute less to the vote.
    Clusters with no labeled members receive the leftover phase letter.
    """
    assigned: dict[int, str] = {}
    remaining = list(_INT_TO_PHASE.values())   # ['A', 'B', 'C']
    labeled   = noisy >= 0

    for c in range(n_clusters):
        mask = (cluster_ids == c) & labeled
        if not mask.any():
            continue
        scores: dict[int, float] = {}
        for pi, w in zip(noisy[mask], probs[mask, c]):
            scores[pi] = scores.get(pi, 0.0) + float(w)
        best = _INT_TO_PHASE[max(scores, key=scores.get)]
        assigned[c] = best
        if best in remaining:
            remaining.remove(best)

    for c in range(n_clusters):
        if c not in assigned:
            assigned[c] = remaining.pop(0) if remaining else "?"

    return assigned


def _log(msg: str) -> None:
    print(f"[phase_cluster] {msg}")


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime, random

    rng = np.random.default_rng(0)
    random.seed(0)

    n_per_phase = 50
    timestamps  = [
        datetime.datetime(2024, 1, 1) + datetime.timedelta(minutes=15 * i)
        for i in range(7 * _SLOTS_PER_DAY)
    ]

    def phase_voltage(slot: int, phase: int, dev_noise: float) -> float:
        base  = 120.0 + phase * 2.0
        daily = 5.0 * np.sin(2 * np.pi * (slot % _SLOTS_PER_DAY) / _SLOTS_PER_DAY + phase * 0.8)
        return round(base + daily + rng.normal(0, 0.5) + dev_noise, 4)

    rows, truth = [], []
    for phase_idx, letter in enumerate(("A", "B", "C")):
        for dev in range(n_per_phase):
            dev_id    = f"DEV_{letter}_{dev:03d}"
            dev_noise = rng.normal(0, 0.3)
            label     = letter if random.random() > 0.05 else random.choice(["A", "B", "C"])
            for slot, ts in enumerate(timestamps):
                rows.append({"PHASE_CHILD": label, "ENDPOINTID": dev_id,
                             "TIMESTAMP": ts, "DATA": phase_voltage(slot, phase_idx, dev_noise)})
            truth.append({"ENDPOINTID": dev_id, "TRUE_PHASE": letter})

    df_input = pd.DataFrame(rows)
    df_truth  = pd.DataFrame(truth)

    df_result = cluster_phase_profiles(df_input)
    validate_clusters(df_result)

    eval_df = (
        df_result
        .drop_duplicates(subset=["ENDPOINTID"])
        [["ENDPOINTID", "PREDICTED_PHASE", "IS_SUSPECT_LABEL"]]
        .merge(df_truth, on="ENDPOINTID")
    )
    correct = (eval_df["PREDICTED_PHASE"] == eval_df["TRUE_PHASE"]).sum()
    total   = len(eval_df)
    print(f"Accuracy: {correct}/{total} ({100 * correct / total:.1f} %)")
    print("\nSuspect-labeled devices:")
    print(eval_df[eval_df["IS_SUSPECT_LABEL"]].to_string(index=False))
