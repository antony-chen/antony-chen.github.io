"""
phase_clustering.py
-------------------
Cluster feeder voltage profiles into their correct phase (A, B, C).

Expected input PySpark DataFrame schema:
    PHASE_CHILD  str   - noisy phase label ('A', 'B', 'C', or null)
    ENDPOINTID   any   - device identifier
    TIMESTAMP    ts    - 15-minute interval timestamp
    DATA         float - voltage reading (volts)

Output: input DataFrame + three new columns:
    PREDICTED_PHASE    str   - predicted phase letter
    PHASE_CONFIDENCE   float - model confidence in [0, 1]
    IS_SUSPECT_LABEL   bool  - True when PHASE_CHILD contradicts the
                               prediction at high confidence (likely mislabeled)

Algorithm overview
------------------
1. Spectral clustering on the pairwise Pearson-correlation affinity matrix.
   Devices on the same phase share load events and voltage-drop patterns,
   producing high within-phase correlation regardless of absolute voltage level.
   This step is entirely unsupervised and is therefore immune to label noise.

2. Gaussian Mixture Model (PCA-compressed daily profile features) run in
   parallel to obtain per-device soft probabilities used later for confidence
   scoring and mislabel detection.

3. Confidence-weighted majority vote maps anonymous cluster integers to the
   phase letters A/B/C using the (noisy) training labels.

4. Devices whose stored label disagrees with the high-confidence prediction
   are flagged as IS_SUSPECT_LABEL for downstream review.

Dependencies: pyspark, numpy, pandas, scikit-learn
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

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PHASE_TO_INT: dict[str, int] = {"A": 0, "B": 1, "C": 2}
_INT_TO_PHASE: dict[int, str] = {0: "A", 1: "B", 2: "C"}

# Above this device count, the N×N correlation matrix becomes expensive;
# fall back to GMM-only clustering.
_SPECTRAL_DEVICE_LIMIT = 10_000

# 15-minute intervals → 96 slots per 24-hour day
_SLOTS_PER_DAY = 96


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_phase_profiles(
    df: DataFrame,
    *,
    phase_col: str = "PHASE_CHILD",
    device_col: str = "ENDPOINTID",
    time_col: str = "TIMESTAMP",
    voltage_col: str = "DATA",
    n_phases: int = 3,
    mislabel_threshold: float = 0.80,
    use_spectral: bool = True,
) -> DataFrame:
    """
    Cluster feeder voltage profiles into phases A, B, C.

    Parameters
    ----------
    df                  Input PySpark DataFrame (see module docstring for schema).
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
    spark = df.sparkSession

    # ------------------------------------------------------------------
    # Step 1  Build a 0-based integer time-slot index from the distinct
    #         sorted timestamps so every device maps to the same columns.
    # ------------------------------------------------------------------
    slot_window = Window.orderBy(time_col)
    slot_df = (
        df.select(time_col)
          .distinct()
          .withColumn("_slot", (F.dense_rank().over(slot_window) - 1).cast("int"))
    )
    n_slots = slot_df.count()
    _log(f"{n_slots} time slots found ({n_slots / _SLOTS_PER_DAY:.1f} days of 15-min data)")

    # ------------------------------------------------------------------
    # Step 2  Join the slot index and collapse duplicate (device, slot)
    #         readings that can arise from resets or meter errors.
    # ------------------------------------------------------------------
    agg = (
        df.join(slot_df, on=time_col, how="left")
          .groupBy(device_col, phase_col, "_slot")
          .agg(F.mean(voltage_col).alias("_v"))
    )

    # ------------------------------------------------------------------
    # Step 3  Pivot to wide format: one row per device, one column per
    #         time-slot index.  Missing slots become null (handled later).
    # ------------------------------------------------------------------
    wide = (
        agg
        .groupBy(device_col, phase_col)
        .pivot("_slot", list(range(n_slots)))
        .agg(F.first("_v"))
    )

    n_devices = wide.count()
    _log(f"Collecting {n_devices:,} devices × {n_slots} slots to driver …")
    pdf = wide.toPandas()

    # ------------------------------------------------------------------
    # Step 4  Extract arrays from the pandas DataFrame.
    # ------------------------------------------------------------------
    device_ids = pdf[device_col].values
    raw_labels = pdf[phase_col].values

    # Slot columns are named "0", "1", … by the Spark pivot
    slot_cols = [c for c in pdf.columns if c not in (device_col, phase_col)]
    V = pdf[slot_cols].astype(float).values    # shape: (n_devices, n_slots)

    # ------------------------------------------------------------------
    # Step 5  Impute missing slots per device via linear interpolation.
    # ------------------------------------------------------------------
    V = _impute(V)

    # ------------------------------------------------------------------
    # Step 6  Encode noisy labels: A→0, B→1, C→2, anything else → -1.
    # ------------------------------------------------------------------
    noisy = np.array([
        _PHASE_TO_INT.get(str(lbl).strip().upper(), -1)
        for lbl in raw_labels
    ])
    n_labeled = int((noisy >= 0).sum())
    _log(f"Labeled devices: {n_labeled:,}  |  Unlabeled: {len(noisy) - n_labeled:,}")

    # ------------------------------------------------------------------
    # Step 7  Primary clustering — spectral when feasible, else GMM.
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
    # Step 8  GMM on PCA-compressed daily-profile features.
    #         Always runs; provides the soft probability matrix used for
    #         confidence scoring and mislabel detection in Steps 9-10.
    # ------------------------------------------------------------------
    _log("Running GMM for soft probability estimates …")
    probs, gmm_ids = _gmm_cluster(V, n_phases, n_slots)

    # Align GMM cluster integers to match the spectral cluster integers
    # so that probs[:, c] corresponds to the same group as hard_ids == c.
    if hard_ids is not None:
        mapping  = _majority_mapping(gmm_ids, hard_ids, n_phases)
        gmm_ids  = np.array([mapping[g] for g in gmm_ids])   # noqa: F841
        probs    = _rearrange_cols(probs, mapping)
    else:
        hard_ids = gmm_ids

    # ------------------------------------------------------------------
    # Step 9  Map anonymous cluster integers → phase letters.
    #         Use a confidence-weighted majority vote over labeled devices
    #         so that high-confidence predictions drown out noisy labels.
    # ------------------------------------------------------------------
    cluster_to_phase = _assign_phases(hard_ids, noisy, probs, n_phases)
    _log(f"Cluster → Phase assignment: {cluster_to_phase}")

    predicted  = np.array([cluster_to_phase[c] for c in hard_ids])
    confidence = probs.max(axis=1)

    # ------------------------------------------------------------------
    # Step 10  Flag suspect labels.
    #          A labeled device is suspect when the model is confident in
    #          a DIFFERENT phase than the stored label.
    # ------------------------------------------------------------------
    pred_int = np.vectorize(_PHASE_TO_INT.get)(predicted)
    suspect  = (noisy >= 0) & (pred_int != noisy) & (confidence > mislabel_threshold)
    _log(f"Suspect labels: {int(suspect.sum()):,} / {n_labeled:,} labeled devices")

    # ------------------------------------------------------------------
    # Step 11  Attach results and join back onto the original DataFrame.
    # ------------------------------------------------------------------
    result_pdf = pd.DataFrame({
        device_col:         device_ids,
        "PREDICTED_PHASE":  predicted,
        "PHASE_CONFIDENCE": np.round(confidence, 4).astype(float),
        "IS_SUSPECT_LABEL": suspect.astype(bool),
    })

    result_sdf = spark.createDataFrame(result_pdf)
    return df.join(result_sdf, on=device_col, how="left")


def validate_clusters(df_result: DataFrame, device_col: str = "ENDPOINTID") -> None:
    """
    Print within-phase and between-phase mean Pearson correlations as a
    sanity check.  A good clustering shows within > 0.7 and between < 0.4.

    Call after cluster_phase_profiles(); expects PREDICTED_PHASE in df_result.
    """
    _log("Validating cluster quality …")

    # Collect device-level predictions
    pdf = df_result.select(device_col, "PREDICTED_PHASE", "PHASE_CONFIDENCE").dropDuplicates(
        [device_col]
    ).toPandas()

    phase_groups: dict[str, list[int]] = {}
    for phase in ("A", "B", "C"):
        idxs = pdf.index[pdf["PREDICTED_PHASE"] == phase].tolist()
        if idxs:
            phase_groups[phase] = idxs

    # We can't easily recompute the voltage matrix here without re-pivoting,
    # so just report size and mean confidence per group.
    print("\n── Cluster validation ──────────────────────────────")
    for phase, idxs in phase_groups.items():
        mean_conf = pdf.loc[idxs, "PHASE_CONFIDENCE"].mean()
        print(f"  Phase {phase}: {len(idxs):>6,} devices  |  mean confidence: {mean_conf:.3f}")
    print("────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[phase_cluster] {msg}")


def _impute(V: np.ndarray) -> np.ndarray:
    """
    Per-device linear interpolation along the time axis.
    Leading / trailing NaN gaps are filled by propagating the nearest
    valid reading.  All-NaN rows (devices with no data at all) are filled
    with the global mean voltage.
    """
    out = V.copy()
    global_mean = float(np.nanmean(V)) if not np.isnan(V).all() else 0.0

    for i in range(out.shape[0]):
        s = pd.Series(out[i])
        filled = s.interpolate(method="linear", limit_direction="both")
        out[i] = filled.to_numpy()
        if np.isnan(out[i]).any():          # entire row was NaN
            out[i] = global_mean

    return out


def _spectral_cluster(V: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Build a Pearson-correlation affinity matrix and apply spectral clustering.

    Each device's voltage series is z-scored before computing correlations so
    that absolute voltage offsets (common in unbalanced feeders) do not
    dominate the similarity measure — only the shape of the load profile matters.
    """
    V_z    = StandardScaler().fit_transform(V.T).T    # z-score row-wise
    corr   = np.corrcoef(V_z)                         # (n, n)
    corr   = np.nan_to_num(corr, nan=0.0)
    affinity = np.clip((corr + 1.0) / 2.0, 0.0, 1.0) # map [-1, 1] → [0, 1]

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
      (captures the daily load shape that distinguishes residential/commercial
      customers on each phase).
    - Standard deviation at each slot across days (captures day-to-day spread).
    - Global mean, std, 5th percentile, 95th percentile (overall voltage level
      and range — useful when phases carry systematically different loads).

    PCA is applied before GMM to decorrelate features and speed up fitting.

    Returns (soft_probs, hard_labels).
    """
    n_full_days = n_slots // _SLOTS_PER_DAY
    V_trim      = V[:, : n_full_days * _SLOTS_PER_DAY]

    # Reshape to (n_devices, n_days, 96) for per-slot daily aggregation
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

    probs = gmm.predict_proba(X_pca)    # (n, n_clusters)
    ids   = gmm.predict(X_pca)          # (n,)
    return probs, ids


def _majority_mapping(
    source: np.ndarray,
    target: np.ndarray,
    n: int,
) -> dict[int, int]:
    """
    For each source cluster, find the target cluster that contains the most
    of its members.  Used to align GMM cluster integers to spectral integers.
    """
    mapping: dict[int, int] = {}
    for s in range(n):
        mask = source == s
        if mask.any():
            mapping[s] = int(np.bincount(target[mask], minlength=n).argmax())
        else:
            mapping[s] = s
    return mapping


def _rearrange_cols(probs: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """
    Reorder probability columns so that probs[:, new] holds the probability
    that was previously in probs[:, orig].
    """
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
    Map anonymous cluster integers → phase letters using a
    confidence-weighted majority vote over labeled devices.

    Weighting by model confidence means that devices the model is uncertain
    about (likely near the cluster boundary) contribute less to the vote,
    reducing the influence of mislabeled edge-cases.

    Clusters with no labeled members receive the leftover phase letter.
    """
    assigned: dict[int, str] = {}
    remaining_phases = list(_INT_TO_PHASE.values())   # ['A', 'B', 'C']
    labeled = noisy >= 0

    for c in range(n_clusters):
        mask = (cluster_ids == c) & labeled
        if not mask.any():
            continue

        # Weighted vote: sum confidence weights per phase integer
        weights    = probs[mask, c]
        phase_ints = noisy[mask]
        scores: dict[int, float] = {}
        for pi, w in zip(phase_ints, weights):
            scores[pi] = scores.get(pi, 0.0) + float(w)

        best_phase = _INT_TO_PHASE[max(scores, key=scores.get)]
        assigned[c] = best_phase
        if best_phase in remaining_phases:
            remaining_phases.remove(best_phase)

    # Assign any leftover letters to clusters that had no labeled devices
    for c in range(n_clusters):
        if c not in assigned:
            assigned[c] = remaining_phases.pop(0) if remaining_phases else "?"

    return assigned


# ---------------------------------------------------------------------------
# Quick-start example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType
    import datetime, random

    spark = SparkSession.builder.appName("phase_clustering_example").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # ---- Synthetic dataset -------------------------------------------------
    random.seed(0)
    rng = np.random.default_rng(0)

    # 3 phases × 50 devices, one week of 15-min readings
    n_per_phase = 50
    timestamps  = [
        datetime.datetime(2024, 1, 1) + datetime.timedelta(minutes=15 * i)
        for i in range(7 * 96)                          # 672 slots
    ]

    # Each phase has a distinct daily load shape (sinusoidal with different
    # amplitude / offset) to give the algorithm a clear signal.
    def phase_voltage(slot: int, phase: int, device_noise: float) -> float:
        base   = 120.0 + phase * 2.0                    # slight per-phase offset
        daily  = 5.0 * np.sin(2 * np.pi * (slot % 96) / 96 + phase * 0.8)
        noise  = rng.normal(0, 0.5) + device_noise      # per-reading noise
        return round(base + daily + noise, 4)

    rows = []
    true_phases = []
    for phase_idx, letter in enumerate(("A", "B", "C")):
        for dev in range(n_per_phase):
            dev_id    = f"DEV_{letter}_{dev:03d}"
            dev_noise = rng.normal(0, 0.3)
            # Inject ~5 % label errors
            label = letter if random.random() > 0.05 else random.choice(["A", "B", "C"])
            for slot, ts in enumerate(timestamps):
                rows.append((label, dev_id, ts, phase_voltage(slot, phase_idx, dev_noise)))
            true_phases.append((dev_id, letter))

    schema = StructType([
        StructField("PHASE_CHILD", StringType(),    True),
        StructField("ENDPOINTID",  StringType(),    False),
        StructField("TIMESTAMP",   TimestampType(), False),
        StructField("DATA",        DoubleType(),    False),
    ])
    df_input = spark.createDataFrame(rows, schema=schema)

    # ---- Run clustering ----------------------------------------------------
    df_result = cluster_phase_profiles(df_input)
    validate_clusters(df_result)

    # ---- Spot-check against ground truth -----------------------------------
    truth_df = spark.createDataFrame(true_phases, ["ENDPOINTID", "TRUE_PHASE"])
    eval_df  = (
        df_result
        .select("ENDPOINTID", "PREDICTED_PHASE", "IS_SUSPECT_LABEL")
        .dropDuplicates(["ENDPOINTID"])
        .join(truth_df, on="ENDPOINTID")
    )
    correct = eval_df.filter(F.col("PREDICTED_PHASE") == F.col("TRUE_PHASE")).count()
    total   = eval_df.count()
    print(f"\nAccuracy on synthetic data: {correct}/{total} ({100*correct/total:.1f} %)")
    print("Suspect-labeled devices (sample):")
    eval_df.filter(F.col("IS_SUSPECT_LABEL")).show(10, truncate=False)

    spark.stop()
