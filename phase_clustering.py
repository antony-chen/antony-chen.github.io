import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

SLOTS_PER_DAY = 96
ENCODE = {"A": 0, "B": 1, "C": 2}
DECODE = {0: "A", 1: "B", 2: "C"}


def cluster_phases(df,
                   phase_col="PHASE_CHILD",
                   device_col="ENDPOINTID",
                   time_col="TIMESTAMP",
                   voltage_col="DATA",
                   n_phases=3,
                   mislabel_threshold=0.80):

    # pivot long → (n_devices × n_slots)
    slots   = {t: i for i, t in enumerate(sorted(df[time_col].unique()))}
    n_slots = len(slots)
    work    = df.copy()
    work["_s"] = work[time_col].map(slots)

    labels = (work.dropna(subset=[phase_col])
              .groupby(device_col)[phase_col]
              .agg(lambda s: s.mode()[0]))

    V_wide = (work.groupby([device_col, "_s"])[voltage_col]
              .mean().unstack("_s").reindex(columns=range(n_slots)))

    ids = V_wide.index.to_numpy()
    V   = V_wide.to_numpy(dtype=float)

    # impute missing slots per device
    gap = np.isnan(V)
    gm  = np.nanmean(V) if not np.isnan(V).all() else 0.0
    for i in range(len(V)):
        V[i] = pd.Series(V[i]).interpolate(limit_direction="both").to_numpy()
        if np.isnan(V[i]).any():
            V[i] = gm

    # encode noisy labels  (unknown → -1)
    noisy = np.array([ENCODE.get(str(labels.get(d, "")).strip().upper(), -1)
                      for d in ids])

    # cluster
    A        = affinity(V, gap)
    clusters = SpectralClustering(n_phases, affinity="precomputed",
                                  n_init=10, random_state=42).fit_predict(A)
    probs     = soft_probs(A, clusters, n_phases)
    phase_map = assign_phases(clusters, noisy, probs, n_phases)

    predicted  = np.array([phase_map[c] for c in clusters])
    confidence = probs.max(axis=1)
    suspect    = (noisy >= 0) & (np.vectorize(ENCODE.get)(predicted) != noisy) \
                              & (confidence > mislabel_threshold)

    result = pd.DataFrame({device_col:        ids,
                           "PREDICTED_PHASE":  predicted,
                           "PHASE_CONFIDENCE": confidence.round(4),
                           "IS_SUSPECT_LABEL": suspect})
    return df.merge(result, on=device_col, how="left")


def affinity(V, gap):
    """
    N×N affinity matrix: 70% first-difference + 30% raw voltage correlation.
    First-difference removes the shared diurnal trend and amplifies
    load-switching events — the strongest per-phase signal at 15-min resolution.
    """
    Vm = V.copy().astype(float)
    Vm[gap] = np.nan
    dV  = np.diff(Vm, axis=1)
    dVz = np.zeros_like(dV)
    for i in range(len(dV)):
        v, ok = dV[i], ~np.isnan(dV[i])
        if ok.sum() > 1:
            s = v[ok].std() or 1.0
            dVz[i] = np.where(ok, (v - v[ok].mean()) / s, 0.0)
    c_dv = np.nan_to_num(np.corrcoef(dVz))

    Vz   = StandardScaler().fit_transform(V.T).T
    c_rv = np.nan_to_num(np.corrcoef(Vz))

    return np.clip((0.7*c_dv + 0.3*c_rv + 1) / 2, 0, 1)


def soft_probs(A, labels, n):
    """
    Soft cluster probabilities from the affinity matrix.
    prob(device i, cluster c) ∝ mean affinity from i to members of c.
    No EM — no collapse risk.
    """
    M   = np.eye(n)[labels]
    raw = (A @ M) / M.sum(axis=0)
    for i in range(len(labels)):
        c, sz = labels[i], M[:, labels[i]].sum()
        if sz > 1:
            raw[i, c] = (raw[i, c]*sz - A[i, i]) / (sz - 1)
    return raw / raw.sum(axis=1, keepdims=True)


def assign_phases(clusters, noisy, probs, n):
    """Confidence-weighted majority vote: cluster integers → phase letters."""
    out, used = {}, []
    for c in range(n):
        mask = (clusters == c) & (noisy >= 0)
        if not mask.any():
            continue
        scores = {}
        for p, w in zip(noisy[mask], probs[mask, c]):
            scores[p] = scores.get(p, 0) + w
        best = DECODE[max(scores, key=scores.get)]
        out[c] = best
        used.append(best)
    for c in range(n):
        if c not in out:
            out[c] = next(p for p in DECODE.values() if p not in used)
            used.append(out[c])
    return out
