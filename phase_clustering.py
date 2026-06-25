import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    V   = V_wide.to_numpy(dtype=float, copy=True)

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


def align_intervals(s1, s2, freq="15min"):
    s1 = s1.copy()
    s2 = s2.copy()
    s1.index = pd.to_datetime(s1.index)
    s2.index = pd.to_datetime(s2.index)
    s1 = s1.resample(freq).mean().dropna()
    s2 = s2.resample(freq).mean().dropna()
    common = s1.index.intersection(s2.index)
    return s1.reindex(common), s2.reindex(common)


def _drop_outliers(s, k=3.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s[(s >= q1 - k * iqr) & (s <= q3 + k * iqr)]


def plot_phase_voltages(df1, df2, phase,
                        time_col="timestamp_cst",
                        voltage_col="DATA",
                        phase_col="PHASE_CHILD",
                        label1="Dataset 1",
                        label2="Dataset 2"):
    d1 = df1[(df1[phase_col] == phase) & df1[voltage_col].notna() & (df1[voltage_col] > 0)]
    d2 = df2[(df2[phase_col] == phase) & df2[voltage_col].notna() & (df2[voltage_col] > 0)]
    s1 = d1.groupby(time_col)[voltage_col].mean().sort_index()
    s2 = d2.groupby(time_col)[voltage_col].mean().sort_index()
    s1 = _drop_outliers(s1)
    s2 = _drop_outliers(s2)
    s1, s2 = align_intervals(s1, s2)

    corr = s1.corr(s2) if len(s1) > 1 else float("nan")

    # rolling correlation to find negatively correlated windows
    window = min(SLOTS_PER_DAY, len(s1))
    neg_periods = []
    if len(s1) >= window:
        rolling_corr = s1.rolling(window, center=True).corr(s2)
        neg_mask = rolling_corr < -0.3
        if neg_mask.any():
            in_run = False
            for idx, is_neg in neg_mask.items():
                if is_neg and not in_run:
                    run_start = idx
                    in_run = True
                elif not is_neg and in_run:
                    neg_periods.append((run_start, idx))
                    in_run = False
            if in_run:
                neg_periods.append((run_start, s1.index[-1]))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(s1.index, s1.values, linewidth=0.8, label=label1, alpha=0.8)
    ax.plot(s2.index, s2.values, linewidth=0.8, label=label2, alpha=0.8)

    for start, end in neg_periods:
        ax.axvspan(start, end, color="red", alpha=0.1)

    ax.set_xlabel("Timestamp (CST)")
    ax.set_ylabel("Voltage")
    ax.set_title(f"Phase {phase} — Pearson r = {corr:.4f}")
    ax.legend()

    all_vals = np.concatenate([s1.values, s2.values])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = (vmax - vmin) * 0.02 or 0.1
    ax.set_ylim(vmin - margin, vmax + margin)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    fig.autofmt_xdate()
    fig.tight_layout()
    ax.set_ylim(vmin - margin, vmax + margin)

    if neg_periods:
        print(f"\n[phase {phase}] {len(neg_periods)} negatively correlated period(s):")
        for i, (start, end) in enumerate(neg_periods, 1):
            seg_corr = s1.loc[start:end].corr(s2.loc[start:end]) if len(s1.loc[start:end]) > 1 else float("nan")
            print(f"  {i}. {start}  →  {end}   (r = {seg_corr:.3f})")
    else:
        print(f"\n[phase {phase}] No negatively correlated periods found.")

    plt.show()
    return corr


def phase_correlation_matrix(df1, df2,
                             time_col="timestamp_cst",
                             voltage_col="DATA",
                             phase_col="PHASE_CHILD",
                             label1="Device 1",
                             label2="Device 2",
                             phases=("A", "B", "C")):
    phases = list(phases)
    s1, s2 = {}, {}
    for p in phases:
        d = df1[(df1[phase_col] == p) & df1[voltage_col].notna() & (df1[voltage_col] > 0)]
        s1[p] = d.groupby(time_col)[voltage_col].mean().sort_index()
        d = df2[(df2[phase_col] == p) & df2[voltage_col].notna() & (df2[voltage_col] > 0)]
        s2[p] = d.groupby(time_col)[voltage_col].mean().sort_index()

    mat = pd.DataFrame(np.nan, index=phases, columns=phases)
    for p1 in phases:
        for p2 in phases:
            a, b = align_intervals(s1[p1], s2[p2])
            if len(a) > 1:
                mat.loc[p1, p2] = a.corr(b)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(phases)))
    ax.set_yticks(range(len(phases)))
    ax.set_xticklabels(phases)
    ax.set_yticklabels(phases)
    ax.set_xlabel(label2)
    ax.set_ylabel(label1)

    for i in range(len(phases)):
        for j in range(len(phases)):
            val = mat.iloc[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.7 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=11, color=color)

    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Phase Correlation: {label1} vs {label2}")
    fig.tight_layout()
    plt.show()
    return mat


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
