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


def _filter_time(df, time_col, start_time, end_time):
    if start_time is None and end_time is None:
        return df
    t = pd.to_datetime(df[time_col])
    mask = pd.Series(True, index=df.index)
    if start_time is not None:
        mask &= t >= pd.Timestamp(start_time)
    if end_time is not None:
        mask &= t <= pd.Timestamp(end_time)
    return df[mask]


def _drop_outliers(s, k=3.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s[(s >= q1 - k * iqr) & (s <= q3 + k * iqr)]


def plot_phase_voltages(df1, df2, phase1, phase2=None,
                        time_col="timestamp_cst",
                        voltage_col="DATA",
                        phase_col="PHASE_CHILD",
                        label1="Dataset 1",
                        label2="Dataset 2",
                        start_time=None,
                        end_time=None):
    if phase2 is None:
        phase2 = phase1
    df1 = _filter_time(df1, time_col, start_time, end_time)
    df2 = _filter_time(df2, time_col, start_time, end_time)
    d1 = df1[(df1[phase_col] == phase1) & df1[voltage_col].notna() & (df1[voltage_col] > 0)]
    d2 = df2[(df2[phase_col] == phase2) & df2[voltage_col].notna() & (df2[voltage_col] > 0)]
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
    phase_label = f"Phase {phase1}" if phase1 == phase2 else f"Phase {phase1} vs {phase2}"
    ax.set_title(f"{phase_label} — Pearson r = {corr:.4f}")
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
        print(f"\n[{phase_label}] {len(neg_periods)} negatively correlated period(s):")
        for i, (start, end) in enumerate(neg_periods, 1):
            seg_corr = s1.loc[start:end].corr(s2.loc[start:end]) if len(s1.loc[start:end]) > 1 else float("nan")
            print(f"  {i}. {start}  →  {end}   (r = {seg_corr:.3f})")
    else:
        print(f"\n[{phase_label}] No negatively correlated periods found.")

    plt.show()
    return corr


def phase_correlation_matrix(df1, df2,
                             time_col="timestamp_cst",
                             voltage_col="DATA",
                             phase_col="PHASE_CHILD",
                             label1="Device 1",
                             label2="Device 2",
                             phases=("A", "B", "C"),
                             start_time=None,
                             end_time=None,
                             csv_path=None,
                             mslink1=None,
                             mslink2=None,
                             feeder=None):
    df1 = _filter_time(df1, time_col, start_time, end_time)
    df2 = _filter_time(df2, time_col, start_time, end_time)
    phases = list(phases)
    s1, s2 = {}, {}
    for p in phases:
        d = df1[(df1[phase_col] == p) & df1[voltage_col].notna() & (df1[voltage_col] > 0)]
        s1[p] = _drop_outliers(d.groupby(time_col)[voltage_col].mean().sort_index())
        d = df2[(df2[phase_col] == p) & df2[voltage_col].notna() & (df2[voltage_col] > 0)]
        s2[p] = _drop_outliers(d.groupby(time_col)[voltage_col].mean().sort_index())

    mat = pd.DataFrame(np.nan, index=phases, columns=phases)
    for p1 in phases:
        for p2 in phases:
            a, b = align_intervals(s1[p1], s2[p2])
            if len(a) > 1:
                mat.loc[p1, p2] = a.corr(b)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat.values.astype(float), cmap="RdYlGn", vmin=-1, vmax=1)
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
    t1 = f"{label1} ({mslink1})" if mslink1 is not None else label1
    t2 = f"{label2} ({mslink2})" if mslink2 is not None else label2
    ax.set_title(f"Phase Correlation: {t1} vs {t2}")
    fig.tight_layout()
    plt.show()

    if csv_path is not None:
        import os
        phase_cols = [f"{p1}-{p2}" for p1 in phases for p2 in phases]
        row = {"mslink_upstream": mslink1, "device_upstream": label1,
               "mslink_downstream": mslink2, "device_downstream": label2,
               "feeder": feeder}
        for col in phase_cols:
            p1, p2 = col.split("-")
            row[col] = mat.loc[p1, p2] if (p1 in mat.index and p2 in mat.columns) else np.nan
        row_df = pd.DataFrame([row])
        if os.path.exists(csv_path):
            existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
            row_df = row_df.reindex(columns=existing_cols)
            row_df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(csv_path, index=False)

    return mat


def analyze_correlations(csv_path, phases=("A", "B", "C")):
    """
    Categorize each device pair in the CSV into one of three groups based on
    how the per-phase correlations align:

    matching   — diagonal dominates: each upstream phase has its highest
                 correlation with the same-named downstream phase.
    mismatched — clear one-to-one bijection, but the labels don't match
                 (e.g. upstream-A correlates best with downstream-C, etc.).
    confusing  — any of:
                   • multiple upstream phases claim the same best downstream phase
                   • max correlation for any upstream phase row is < 0.5
                   • any matrix cell is missing

    Returns a dict with keys "matching", "mismatched", "confusing", each
    holding a DataFrame of the corresponding pairs.
    """
    df = pd.read_csv(csv_path)
    phases = list(phases)

    matching_rows, mismatched_rows, confusing_rows = [], [], []

    for _, row in df.iterrows():
        up_mslink   = row.get("mslink_upstream")
        down_mslink = row.get("mslink_downstream")
        up   = f"{row.get('device_upstream',   '?')} ({up_mslink})"
        down = f"{row.get('device_downstream', '?')} ({down_mslink})"

        base = {
            "mslink_upstream":   up_mslink,
            "device_upstream":   row.get("device_upstream"),
            "mslink_downstream": down_mslink,
            "device_downstream": row.get("device_downstream"),
            "feeder":            row.get("feeder"),
        }

        # build 3×3 matrix; flag any missing cells
        mat, missing = {}, False
        for p1 in phases:
            for p2 in phases:
                val = row.get(f"{p1}-{p2}", float("nan"))
                mat[(p1, p2)] = float(val) if not pd.isna(val) else float("nan")
                if pd.isna(mat[(p1, p2)]):
                    missing = True

        if missing:
            confusing_rows.append({**base, "reason": "missing phase data"})
            print(f"[CONFUSING]  {up}  →  {down}  (missing phase data)")
            continue

        # for each upstream phase, find which downstream phase it correlates best with
        mapping, confusing, reason = {}, False, ""
        for p1 in phases:
            row_vals = {p2: mat[(p1, p2)] for p2 in phases}
            best_p2  = max(row_vals, key=row_vals.get)
            best_val = row_vals[best_p2]
            if best_val < 0.5:
                confusing = True
                reason = f"low max correlation for phase {p1} (r={best_val:.3f})"
                break
            mapping[p1] = best_p2

        if not confusing:
            claimed = list(mapping.values())
            dups = [p for p in phases if claimed.count(p) > 1]
            if dups:
                confusing = True
                reason = f"phases {dups} claimed by multiple upstream phases"

        mapping_str = "  ".join(f"{p}→{mapping.get(p, '?')}" for p in phases)

        if confusing:
            confusing_rows.append({**base, "reason": reason})
            print(f"[CONFUSING]  {up}  →  {down}  ({reason})")
        elif all(mapping[p] == p for p in phases):
            matching_rows.append({**base, "mapping": mapping_str})
            print(f"[MATCHING]   {up}  →  {down}  ({mapping_str})")
        else:
            mismatched_rows.append({**base, "mapping": mapping_str})
            print(f"[MISMATCHED] {up}  →  {down}  ({mapping_str})")

    total = len(matching_rows) + len(mismatched_rows) + len(confusing_rows)
    pct = lambda n, d: f" ({100*n/d:.1f}%)" if d else ""
    print(f"\n── Summary ─────────────────────────────────────────────")
    print(f"  Total pairs:    {total}")
    print(f"  Matching:       {len(matching_rows)}{pct(len(matching_rows), total)}")
    print(f"  Mismatched:     {len(mismatched_rows)}{pct(len(mismatched_rows), total)}")
    print(f"  Confusing:      {len(confusing_rows)}{pct(len(confusing_rows), total)}")

    # per-feeder breakdown
    all_rows = (
        [{"feeder": r.get("feeder"), "cat": "matching"}   for r in matching_rows]
      + [{"feeder": r.get("feeder"), "cat": "mismatched"} for r in mismatched_rows]
      + [{"feeder": r.get("feeder"), "cat": "confusing"}  for r in confusing_rows]
    )
    feeder_df = pd.DataFrame(all_rows)
    feeders = sorted(feeder_df["feeder"].dropna().unique())
    if feeders:
        print(f"\n  Per-feeder breakdown:")
        col_w = max(len(str(f)) for f in feeders)
        header = f"  {'Feeder':<{col_w}}   {'Total':>5}   {'Matching':>8}   {'Mismatched':>10}   {'Confusing':>9}"
        print(f"  {header}")
        print(f"  {'-' * (len(header) - 2)}")
        for feeder in feeders:
            sub = feeder_df[feeder_df["feeder"] == feeder]
            n   = len(sub)
            nm  = (sub["cat"] == "matching").sum()
            nw  = (sub["cat"] == "mismatched").sum()
            nc  = (sub["cat"] == "confusing").sum()
            print(f"  {str(feeder):<{col_w}}   {n:>5}"
                  f"   {nm:>4}{pct(nm, n):>5}"
                  f"   {nw:>6}{pct(nw, n):>5}"
                  f"   {nc:>5}{pct(nc, n):>5}")

        # write a single device CSV for all feeders with no mismatched pairs
        from pathlib import Path
        input_path = Path(csv_path)
        clean_feeders = [f for f in feeders
                         if (feeder_df[feeder_df["feeder"] == f]["cat"] == "mismatched").sum() == 0]
        if clean_feeders:
            pieces = []
            for feeder in clean_feeders:
                feeder_src = df[df["feeder"] == feeder]
                ups   = feeder_src[["mslink_upstream",   "device_upstream"]].rename(
                            columns={"mslink_upstream":   "mslink", "device_upstream":   "device"})
                downs = feeder_src[["mslink_downstream", "device_downstream"]].rename(
                            columns={"mslink_downstream": "mslink", "device_downstream": "device"})
                pieces.append(pd.concat([ups, downs]).assign(feeder=feeder))
            devices = (pd.concat(pieces)
                         .dropna(subset=["mslink"])
                         .drop_duplicates("mslink")
                         .reset_index(drop=True))
            out = input_path.parent / f"{input_path.stem}__devices.csv"
            devices.to_csv(out, index=False)
            print(f"\n  {len(devices)} devices from {len(clean_feeders)} clean feeder(s) "
                  f"({', '.join(str(f) for f in clean_feeders)})  →  {out.name}")
    print(f"────────────────────────────────────────────────────────")

    return {
        "matching":   pd.DataFrame(matching_rows),
        "mismatched": pd.DataFrame(mismatched_rows),
        "confusing":  pd.DataFrame(confusing_rows),
    }


def rank_phase_mismatches(csv_path, phases=("A", "B", "C")):
    """
    Rank unexpected device pairs by mismatch confidence.

    Confidence is the mean per-phase excess — for each upstream phase p,
    how much does the best cross-phase correlation beat the matching one:

        excess(p) = max(corr(p-q) for q != p) - corr(p-p)

    confidence = mean(excess(p)) across phases where corr(p-p) is available.

    A high positive value means the cross-phase signal dominates the diagonal
    strongly and consistently, giving high confidence that the labels are wrong.
    The suggested re-mapping shows what the data actually implies for each phase.
    """
    df = pd.read_csv(csv_path)
    phases = list(phases)

    cross_pairs = [(f"{p1}-{p2}", p1, p2) for p1 in phases for p2 in phases if p1 != p2]

    records = []
    for _, row in df.iterrows():
        # build the correlation matrix for this pair from CSV columns
        mat = {}
        for p1 in phases:
            for p2 in phases:
                val = row.get(f"{p1}-{p2}", float("nan"))
                mat[(p1, p2)] = float(val) if not pd.isna(val) else float("nan")

        # skip pairs with no unexpected correlations (same filter as analyze_correlations)
        is_unexpected = any(
            not np.isnan(mat[(p1, p2)]) and
            not np.isnan(mat[(p1, p1)]) and
            not np.isnan(mat[(p2, p2)]) and
            (mat[(p1, p2)] > mat[(p1, p1)] or mat[(p1, p2)] > mat[(p2, p2)])
            for _, p1, p2 in cross_pairs
        )
        if not is_unexpected:
            continue

        # per-phase excess and suggested true mapping
        excesses = []
        suggested_map = {}
        for p1 in phases:
            diag = mat.get((p1, p1), float("nan"))
            if np.isnan(diag):
                continue
            cross_vals = {p2: mat.get((p1, p2), float("nan"))
                          for p2 in phases if p2 != p1}
            best_p2  = max(cross_vals, key=lambda p: cross_vals[p]
                           if not np.isnan(cross_vals[p]) else -np.inf)
            best_val = cross_vals[best_p2]
            if np.isnan(best_val):
                continue
            excess = best_val - diag
            excesses.append(excess)
            suggested_map[p1] = best_p2 if excess > 0 else p1

        if not excesses:
            continue

        confidence = float(np.mean(excesses))
        mapping_str = "  ".join(f"{p}→{suggested_map.get(p, p)}" for p in phases)

        records.append({
            "mslink_upstream":   row.get("mslink_upstream"),
            "device_upstream":   row.get("device_upstream"),
            "mslink_downstream": row.get("mslink_downstream"),
            "device_downstream": row.get("device_downstream"),
            "feeder":            row.get("feeder"),
            "confidence":        round(confidence, 4),
            "suggested_mapping": mapping_str,
        })

    if not records:
        print("No unexpected pairs found in CSV.")
        return pd.DataFrame()

    result = (pd.DataFrame(records)
                .sort_values("confidence", ascending=False)
                .reset_index(drop=True))

    print("── Phase Mismatch Rankings ─────────────────────────────────────────────")
    for i, r in result.iterrows():
        up   = f"{r['device_upstream']} ({r['mslink_upstream']})"
        down = f"{r['device_downstream']} ({r['mslink_downstream']})"
        print(f"  #{i+1:>3}  confidence={r['confidence']:+.4f}  {up}  →  {down}")
        print(f"         suggested: {r['suggested_mapping']}")
    print("────────────────────────────────────────────────────────────────────────")

    return result


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
