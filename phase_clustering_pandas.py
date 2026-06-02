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
1.  Pivot the long-format DataFrame into a wide voltage matrix
    (n_devices × n_time_slots) and linearly interpolate any gaps.

2.  Spectral clustering on the pairwise Pearson-correlation affinity matrix
    of the FIRST-DIFFERENCE (step-change) series — dV[t] = V[t] - V[t-1].
    Differencing removes the fleet-wide diurnal common-mode signal and
    amplifies load-event signals that are phase-specific.  Devices on the
    same phase share load events; cross-phase pairs are uncorrelated.

3.  Gaussian Mixture Model on phase-discriminative engineered features:
      • Fleet-residual daily profile  (remove the common diurnal pattern)
      • First-difference statistics   (step-change volatility and skew)
      • High-load window features     (voltage during peak-demand slots)
      • Voltage event features        (count/timing of large step changes)
      • Peak timing                   (when in the day voltage is lowest)
      • FFT phase angle               (daily-cycle phase offset per device)
      • Weekday/weekend differential  (fleet-normalised)
    The GMM is warm-started from the spectral cluster labels, preventing the
    EM algorithm from collapsing to a single dominant component.

4.  Confidence-weighted majority vote maps anonymous cluster integers to
    phase letters A/B/C using the (noisy) training labels.

5.  Devices whose stored label disagrees with the high-confidence prediction
    are flagged as IS_SUSPECT_LABEL for downstream review.

Feature-engineering rationale (from phase-identification literature)
---------------------------------------------------------------------
The root cause of GMM collapsing to one cluster is that raw daily-mean
features are dominated by the common diurnal pattern shared by all phases
(everyone uses electricity in the morning and evening). The between-phase
variance is tiny relative to this common-mode signal, so GMM sees one
unimodal distribution and places ~all devices in one Gaussian.

The remedies applied here, each supported by published phase-ID research:

  a) FLEET RESIDUAL   Subtract the fleet-wide median daily profile from each
                      device's profile. This zeros out the common component
                      V_common(t) and leaves only V_phase(t) + noise, which
                      is the actual phase-membership signal.

  b) FIRST DIFFERENCES dV[t] = V[t]-V[t-1] acts as a high-pass filter that
                      removes slow shared trends and amplifies load-switching
                      events. The correlation of dV series between same-phase
                      devices is much stronger than raw-voltage correlation.

  c) HIGH-LOAD WINDOWS During peak demand, each phase's voltage drop is
                      proportional to its own load current. The discriminative
                      SNR is highest during these periods; low-load periods
                      (when all phase voltages are near nominal) are filtered
                      out from the high-load features.

  d) FFT PHASE ANGLE   The magnitude of the daily Fourier harmonic is the
                      same for all phases. The ANGLE differs if loads on
                      different phases peak at slightly different times.

  e) GMM WARM-START   Initialising GMM means from spectral cluster centroids
                      bypasses the EM saddle point that causes collapse.

References
----------
- IEEE 2018 doi:10.1109/TSG.2018.2866842   (PCC-based phase clustering)
- IEEE 2019 doi:10.1109/TSG.2019.2914346   (spectral clustering for phases)
- arXiv 2111.10500  (high-load window / data segmentation approach)
- arXiv 2212.12650  (Fourier compression for phase clustering)
- arXiv 2204.06372  (survey of low-voltage phase identification methods)

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
from sklearn.preprocessing import RobustScaler

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
    # Step 1  Assign a 0-based integer time-slot index from sorted
    #         distinct timestamps.
    # ------------------------------------------------------------------
    sorted_times = sorted(df[time_col].unique())
    time_to_slot = {t: i for i, t in enumerate(sorted_times)}
    n_slots = len(sorted_times)
    _log(f"{n_slots} time slots ({n_slots / _SLOTS_PER_DAY:.1f} days of 15-min data)")

    # ------------------------------------------------------------------
    # Step 2  Collapse duplicate (device, timestamp) readings by
    #         averaging, keep one phase label per device (majority vote),
    #         then pivot to wide format.
    # ------------------------------------------------------------------
    df_work = df[[device_col, phase_col, time_col, voltage_col]].copy()
    df_work["_slot"] = df_work[time_col].map(time_to_slot)

    device_labels = (
        df_work.dropna(subset=[phase_col])
        .groupby(device_col)[phase_col]
        .agg(lambda s: s.mode().iloc[0] if not s.empty else np.nan)
    )

    agg = (
        df_work
        .groupby([device_col, "_slot"], sort=False)[voltage_col]
        .mean()
        .reset_index()
    )
    V_wide = agg.pivot(index=device_col, columns="_slot", values=voltage_col)
    V_wide = V_wide.reindex(columns=range(n_slots))

    n_devices = len(V_wide)
    _log(f"Voltage matrix: {n_devices:,} devices × {n_slots} slots")

    # ------------------------------------------------------------------
    # Step 3  Extract arrays.
    # ------------------------------------------------------------------
    device_ids = V_wide.index.to_numpy()
    V          = V_wide.to_numpy(dtype=float)
    raw_labels = device_labels.reindex(device_ids).to_numpy()

    # ------------------------------------------------------------------
    # Step 4  Impute missing slots.
    #         NOTE: imputed segments are NOT used in the first-difference
    #         correlation (see _spectral_cluster) to avoid inflating
    #         correlation with artificial smooth segments.
    # ------------------------------------------------------------------
    V_imputed, gap_mask = _impute(V)

    # ------------------------------------------------------------------
    # Step 5  Encode noisy labels.
    # ------------------------------------------------------------------
    noisy = np.array([
        _PHASE_TO_INT.get(str(lbl).strip().upper(), -1)
        if pd.notna(lbl) else -1
        for lbl in raw_labels
    ])
    n_labeled = int((noisy >= 0).sum())
    _log(f"Labeled devices: {n_labeled:,}  |  Unlabeled: {len(noisy) - n_labeled:,}")

    # ------------------------------------------------------------------
    # Step 6  Primary spectral clustering on first-difference correlation.
    #         Using dV instead of raw V removes common-mode and amplifies
    #         load-event signals (the key phase-membership signal).
    # ------------------------------------------------------------------
    if use_spectral and n_devices <= _SPECTRAL_DEVICE_LIMIT:
        _log("Running spectral clustering on first-difference correlation affinity …")
        hard_ids = _spectral_cluster(V_imputed, gap_mask, n_phases)
    else:
        if use_spectral:
            _log(
                f"Device count {n_devices:,} > limit {_SPECTRAL_DEVICE_LIMIT:,}"
                " — skipping spectral, using GMM hard labels"
            )
        hard_ids = None

    # ------------------------------------------------------------------
    # Step 7  GMM on phase-discriminative engineered features,
    #         warm-started from spectral labels to prevent EM collapse.
    # ------------------------------------------------------------------
    _log("Running GMM with phase-discriminative features …")
    probs, gmm_ids = _gmm_cluster(V_imputed, n_phases, n_slots, spectral_labels=hard_ids)

    # Align GMM cluster integers to spectral cluster integers.
    if hard_ids is not None:
        mapping = _majority_mapping(gmm_ids, hard_ids, n_phases)
        gmm_ids = np.array([mapping[g] for g in gmm_ids])   # noqa: F841
        probs   = _rearrange_cols(probs, mapping)
    else:
        hard_ids = gmm_ids

    # ------------------------------------------------------------------
    # Step 8  Map cluster integers → phase letters via weighted majority
    #         vote over labeled devices.
    # ------------------------------------------------------------------
    cluster_to_phase = _assign_phases(hard_ids, noisy, probs, n_phases)
    _log(f"Cluster → Phase assignment: {cluster_to_phase}")

    predicted  = np.array([cluster_to_phase[c] for c in hard_ids])
    confidence = probs.max(axis=1)

    # ------------------------------------------------------------------
    # Step 9  Flag suspect labels.
    # ------------------------------------------------------------------
    pred_int = np.vectorize(_PHASE_TO_INT.get)(predicted)
    suspect  = (noisy >= 0) & (pred_int != noisy) & (confidence > mislabel_threshold)
    _log(f"Suspect labels: {int(suspect.sum()):,} / {n_labeled:,} labeled devices")

    # ------------------------------------------------------------------
    # Step 10  Merge results back onto the original DataFrame.
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
    A good clustering shows roughly balanced counts and mean confidence > 0.80.
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
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(V: np.ndarray, n_slots: int) -> np.ndarray:
    """
    Build a phase-discriminative feature matrix from the voltage array.

    All features are fleet-normalised (fleet median subtracted) so the
    dominant common-mode diurnal signal does not overwhelm the phase signal.

    Parameters
    ----------
    V        Voltage matrix, shape (n_devices, n_slots), already imputed.
    n_slots  Total number of time slots.

    Returns
    -------
    Feature matrix of shape (n_devices, n_features).
    """
    n = V.shape[0]
    n_full_days = n_slots // _SLOTS_PER_DAY
    V_trim  = V[:, : n_full_days * _SLOTS_PER_DAY]             # (n, days*96)
    V_by_day = V_trim.reshape(n, n_full_days, _SLOTS_PER_DAY)  # (n, days, 96)

    # ── 1. FLEET-RESIDUAL DAILY PROFILE ────────────────────────────────
    # Subtract the fleet-wide median daily profile to remove the common
    # diurnal pattern. What remains is the phase-specific deviation: the
    # differential voltage drop caused by unequal loading on each phase.
    daily_mean         = V_by_day.mean(axis=1)                  # (n, 96)
    daily_std          = V_by_day.std(axis=1)                   # (n, 96)
    fleet_daily_median = np.median(daily_mean, axis=0)          # (96,)  common mode
    fleet_std_median   = np.median(daily_std,  axis=0)          # (96,)
    residual_profile   = daily_mean - fleet_daily_median        # (n, 96) phase-specific
    residual_std       = daily_std  - fleet_std_median          # (n, 96)

    # ── 2. FIRST-DIFFERENCE STATISTICS ─────────────────────────────────
    # dV[t] = V[t] - V[t-1] acts as a high-pass filter: slow shared trends
    # cancel and load-switching events (the strongest phase signal) are
    # amplified.  Step-change series of same-phase devices are highly
    # correlated; cross-phase pairs are near-zero.
    dV      = np.diff(V, axis=1)          # (n, n_slots-1)
    dV_abs  = np.abs(dV)
    dV_mean = dV_abs.mean(axis=1)         # mean volatility per device
    dV_std  = dV.std(axis=1)              # spread of step changes
    dV_p95  = np.percentile(dV_abs, 95, axis=1)  # magnitude of large events

    # Skewness: negative skew = more drops than rises (heavy-load device)
    dV_skew = _skewness(dV)

    # ── 3. HIGH-LOAD WINDOW FEATURES ───────────────────────────────────
    # Phase voltage drops are largest during peak demand — this is when
    # the SNR for phase discrimination is highest.  Select the 30 % of
    # time slots where the fleet-average voltage is lowest (= peak load).
    fleet_avg       = V.mean(axis=0)                            # (n_slots,)
    high_load_mask  = fleet_avg <= np.percentile(fleet_avg, 30) # bool (n_slots,)
    n_hl            = high_load_mask.sum()

    if n_hl > 0:
        V_hl         = V[:, high_load_mask]                     # (n, n_hl)
        hl_mean      = V_hl.mean(axis=1)
        hl_std       = V_hl.std(axis=1)
        # Fleet-normalise: the deviation from the median device is the signal
        hl_residual  = hl_mean - np.median(hl_mean)
    else:
        hl_mean     = V.mean(axis=1)
        hl_std      = V.std(axis=1)
        hl_residual = np.zeros(n)

    # ── 4. VOLTAGE EVENT FEATURES ──────────────────────────────────────
    # A large load switching on/off causes a step change that all devices
    # on the same phase see simultaneously.  Count and characterise these
    # events.  Events are defined as |dV| > mean + 2σ (global threshold).
    event_thresh = dV_abs.mean() + 2.0 * dV_abs.std()
    n_drops = (dV < -event_thresh).sum(axis=1).astype(float)
    n_rises = (dV >  event_thresh).sum(axis=1).astype(float)

    # Distribute events across four 6-hour windows of the day to capture
    # WHEN each device experiences load-switching activity.
    quarter = max(1, n_slots // 4)
    event_by_quarter = np.column_stack([
        (np.abs(dV[:, i * quarter : (i + 1) * quarter]) > event_thresh).sum(axis=1)
        for i in range(4)
    ])  # (n, 4)

    # ── 5. PEAK TIMING ─────────────────────────────────────────────────
    # If different phases serve customers with different daily routines
    # (e.g., more commercial vs residential), their voltage minima fall
    # at different times of day.  Normalise to fraction of day [0, 1].
    peak_load_slot = (daily_mean.argmin(axis=1) / _SLOTS_PER_DAY).astype(float)
    peak_gen_slot  = (daily_mean.argmax(axis=1) / _SLOTS_PER_DAY).astype(float)

    # ── 6. FFT PHASE ANGLE OF DAILY HARMONIC ───────────────────────────
    # The magnitude of the 24-hour Fourier component is nearly the same
    # for all phases (common-mode again).  The ANGLE differs when loads on
    # different phases peak at slightly different times of day.
    # Encode as (cos, sin) to avoid discontinuity at ±π.
    fft = np.fft.rfft(V_trim, axis=1)                           # (n, freq_bins)
    daily_k = n_full_days  # index of the 1-cycle-per-day harmonic
    if daily_k < fft.shape[1]:
        angle          = np.angle(fft[:, daily_k])
        fft_daily_cos  = np.cos(angle)
        fft_daily_sin  = np.sin(angle)
        fft_daily_mag  = np.abs(fft[:, daily_k]) - np.median(np.abs(fft[:, daily_k]))
    else:
        fft_daily_cos = fft_daily_sin = fft_daily_mag = np.zeros(n)

    # Top-magnitude frequency components (exclude DC at index 0)
    fft_magnitudes = np.abs(fft[:, 1:15])                       # (n, 14)

    # ── 7. WEEKDAY / WEEKEND DIFFERENTIAL ──────────────────────────────
    # Commercial-heavy phases show a larger weekday–weekend voltage spread
    # than residential-heavy phases.  Fleet-normalise to remove the global
    # weekday/weekend effect shared by all phases.
    if n_full_days >= 7:
        wd_mean  = V_by_day[:, :5, :].mean(axis=(1, 2))
        we_mean  = V_by_day[:, 5:7, :].mean(axis=(1, 2))
        wd_we    = (wd_mean - we_mean) - np.median(wd_mean - we_mean)
    else:
        wd_we = np.zeros(n)

    # ── 8. FLEET-NORMALISED SUMMARY STATISTICS ─────────────────────────
    global_mean = V.mean(axis=1)
    summary = np.column_stack([
        global_mean          - np.median(global_mean),
        V.std(axis=1),
        np.percentile(V,  5, axis=1) - np.median(np.percentile(V,  5, axis=1)),
        np.percentile(V, 95, axis=1) - np.median(np.percentile(V, 95, axis=1)),
    ])

    return np.column_stack([
        residual_profile,    # 96  — fleet-normalised daily shape  ← KEY
        residual_std,        # 96  — fleet-normalised daily volatility
        dV_mean,             #  1  — mean step-change magnitude
        dV_std,              #  1  — spread of step changes
        dV_p95,              #  1  — 95th-pct step-change magnitude
        dV_skew,             #  1  — asymmetry (more drops vs rises)
        hl_residual,         #  1  — high-load fleet-residual mean voltage
        hl_std,              #  1  — volatility during peak-demand
        n_drops,             #  1  — count of large voltage drops
        n_rises,             #  1  — count of large voltage rises
        event_by_quarter,    #  4  — event timing across 6-hour windows
        peak_load_slot,      #  1  — daily voltage-minimum timing
        peak_gen_slot,       #  1  — daily voltage-maximum timing
        fft_daily_cos,       #  1  — daily harmonic phase angle (cos)
        fft_daily_sin,       #  1  — daily harmonic phase angle (sin)
        fft_daily_mag,       #  1  — daily harmonic magnitude (fleet-normalised)
        fft_magnitudes,      # 14  — top FFT components
        wd_we,               #  1  — weekday/weekend differential
        summary,             #  4  — fleet-normalised summary stats
                             # ─────────────────────────────────────────
                             # TOTAL: ~226 features before PCA
    ])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[phase_cluster] {msg}")


def _impute(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-device linear interpolation along the time axis.

    Returns
    -------
    V_filled   Imputed voltage matrix.
    gap_mask   Boolean (n_devices, n_slots) — True where a value was missing.
               Used downstream to exclude imputed segments from first-difference
               correlations so artificial smoothness doesn't inflate similarity.
    """
    out      = V.copy()
    gap_mask = np.isnan(V)
    global_mean = float(np.nanmean(V)) if not np.isnan(V).all() else 0.0

    for i in range(out.shape[0]):
        s = pd.Series(out[i])
        out[i] = s.interpolate(method="linear", limit_direction="both").to_numpy()
        if np.isnan(out[i]).any():
            out[i] = global_mean

    return out, gap_mask


def _spectral_cluster(
    V: np.ndarray,
    gap_mask: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Spectral clustering on the pairwise Pearson correlation of FIRST
    DIFFERENCES (step-change series).

    Using dV rather than raw V:
      • Removes the large common-mode diurnal trend shared by all phases.
      • Amplifies load-switching events — the strongest phase-membership
        signal available at 15-minute resolution.
      • Means same-phase devices show much higher mutual correlation, and
        cross-phase pairs are close to zero.

    Imputed segments are masked to NaN before differencing so that
    artificially smooth interpolated runs do not inflate correlation.
    """
    # Mask imputed values before differencing
    V_masked = V.astype(float).copy()
    V_masked[gap_mask] = np.nan

    dV = np.diff(V_masked, axis=1)    # (n, n_slots-1), NaN where gap boundary

    # For each device z-score its own step-change series, ignoring NaN
    dV_z = np.zeros_like(dV)
    for i in range(dV.shape[0]):
        row   = dV[i]
        valid = ~np.isnan(row)
        if valid.sum() > 1:
            mu  = row[valid].mean()
            sig = row[valid].std()
            dV_z[i] = np.where(valid, (row - mu) / (sig if sig > 0 else 1.0), 0.0)

    # Pairwise correlation — use numpy masked array to handle remaining NaNs
    dV_z = np.nan_to_num(dV_z, nan=0.0)
    corr  = np.corrcoef(dV_z)
    corr  = np.nan_to_num(corr, nan=0.0)

    # Also blend in raw-voltage correlation (30 % weight) for stability
    # when the step-change signal is weak (lightly loaded feeder).
    from sklearn.preprocessing import StandardScaler
    V_z      = StandardScaler().fit_transform(V.T).T
    corr_raw = np.corrcoef(V_z)
    corr_raw = np.nan_to_num(corr_raw, nan=0.0)

    corr_blended = 0.7 * corr + 0.3 * corr_raw
    affinity = np.clip((corr_blended + 1.0) / 2.0, 0.0, 1.0)

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
    spectral_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GMM on phase-discriminative features, optionally warm-started from
    spectral cluster labels to prevent EM collapse.

    Why warm-starting prevents the 90 % one-cluster collapse
    --------------------------------------------------------
    GMM uses EM, which is sensitive to initialisation. When features are
    even slightly non-separable (as raw voltage features are), EM slides
    toward a degenerate solution: one large Gaussian swallows the data and
    the other two components shrink to zero weight. Providing spectral
    cluster centroids as initial means places EM in the right basin of
    attraction from the start, so it converges to the physically correct
    three-component solution instead of the degenerate one.

    Parameters
    ----------
    spectral_labels  Hard cluster assignments from _spectral_cluster, or
                     None to use k-means initialisation.

    Returns
    -------
    (soft_probs, hard_labels)
    """
    features = _extract_features(V, n_slots)

    # RobustScaler (median/IQR) is more resistant to voltage outliers
    # than StandardScaler (mean/std).
    X     = RobustScaler().fit_transform(features)
    n_pca = min(30, X.shape[1], X.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)

    # ── Warm-start: compute initial means from spectral centroids ──────
    if spectral_labels is not None:
        init_means = np.array([
            X_pca[spectral_labels == c].mean(axis=0)
            if (spectral_labels == c).any()
            else X_pca.mean(axis=0)
            for c in range(n_clusters)
        ])
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="tied",   # shared covariance: prevents one component
            means_init=init_means,    # from expanding to cover everything
            n_init=5,
            random_state=42,
        )
    else:
        # No spectral labels — fall back to multiple k-means initialisations
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="tied",
            n_init=20,
            random_state=42,
        )

    gmm.fit(X_pca)
    probs = gmm.predict_proba(X_pca)
    ids   = gmm.predict(X_pca)
    return probs, ids


def _skewness(X: np.ndarray) -> np.ndarray:
    """Row-wise skewness: (mean of cubed deviations) / std^3."""
    mu   = X.mean(axis=1, keepdims=True)
    sig  = X.std(axis=1, keepdims=True)
    sig  = np.where(sig == 0, 1.0, sig)
    return ((X - mu) ** 3).mean(axis=1) / (sig.squeeze() ** 3)


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
    """
    assigned: dict[int, str] = {}
    remaining = list(_INT_TO_PHASE.values())
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

    n_per_phase = 50
    timestamps  = [
        datetime.datetime(2024, 1, 1) + datetime.timedelta(minutes=15 * i)
        for i in range(7 * _SLOTS_PER_DAY)
    ]

    def phase_voltage(slot: int, phase: int, dev_noise: float) -> float:
        base  = 120.0 + phase * 2.0
        daily = 5.0 * np.sin(2 * np.pi * (slot % _SLOTS_PER_DAY) / _SLOTS_PER_DAY + phase * 0.8)
        noise = rng.normal(0, 0.5) + dev_noise
        return round(base + daily + noise, 4)

    rows: list[dict] = []
    ground_truth: list[dict] = []

    for phase_idx, letter in enumerate(("A", "B", "C")):
        for dev in range(n_per_phase):
            dev_id    = f"DEV_{letter}_{dev:03d}"
            dev_noise = rng.normal(0, 0.3)
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
    print(f"Accuracy on synthetic data: {correct}/{total} ({100 * correct / total:.1f} %)")
    print("\nSuspect-labeled devices (sample):")
    print(eval_df[eval_df["IS_SUSPECT_LABEL"]].head(10).to_string(index=False))
