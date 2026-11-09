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


def align_intervals(s1, s2, freq="15min", min_points=2):
    s1 = s1.copy()
    s2 = s2.copy()

    def _to_naive(idx):
        idx = pd.to_datetime(idx)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return idx

    s1.index = _to_naive(s1.index)
    s2.index = _to_naive(s2.index)
    s1 = s1.resample(freq).mean().dropna()
    s2 = s2.resample(freq).mean().dropna()
    common = s1.index.intersection(s2.index)
    if len(common) < min_points:
        empty = pd.Series(dtype=float)
        return empty, empty
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


def _corr(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = float(a.corr(b))
    if np.isnan(result):
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        # constant series — same value means identical, different means no relationship
        a_const = a.std(ddof=1) == 0
        b_const = b.std(ddof=1) == 0
        if a_const and b_const:
            return 1.0 if float(a.iloc[0]) == float(b.iloc[0]) else 0.0
        return 0.0
    return result


def voltage_correlation_matrix(devices,
                               time_col="timestamp_cst",
                               voltage_col="DATA",
                               start_time=None,
                               end_time=None):
    """
    Compute a pairwise Pearson correlation matrix across device voltage profiles.

    devices : list of (df, label) tuples
        df    — DataFrame already filtered to the desired phase
        label — name for the row/column in the output matrix

    Returns a square DataFrame of correlation coefficients, or None if
    fewer than 2 series could be built.
    """
    series_map = {}

    for df, label in devices:
        df = _filter_time(df, time_col, start_time, end_time)
        d = df[df[voltage_col].notna() & (df[voltage_col] > 0)]
        if d.empty:
            continue
        s = _drop_outliers(d.groupby(time_col)[voltage_col].mean().sort_index())
        if s.empty:
            continue
        idx = pd.to_datetime(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        s.index = idx
        s = s.resample("15min").mean().dropna()
        if s.empty:
            continue
        series_map[label] = s

    if len(series_map) < 2:
        print("Need at least 2 non-empty series to compute correlations.")
        return None

    combined = pd.concat(series_map.values(), axis=1, keys=series_map.keys())
    labels = list(series_map.keys())
    n = len(labels)
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix.iloc[i, j] = 1.0
            elif j < i:
                matrix.iloc[i, j] = matrix.iloc[j, i]
            else:
                a = combined[labels[i]].dropna()
                b = combined[labels[j]].dropna()
                common = a.index.intersection(b.index)
                matrix.iloc[i, j] = _corr(a.reindex(common), b.reindex(common))

    fig, ax = plt.subplots(figsize=(max(4, n), max(3, n - 1)))
    vals = matrix.values.astype(float)
    im = ax.imshow(vals, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = vals[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(v) < 0.7 else "white")
    ax.set_title("Voltage Profile Correlations")
    fig.tight_layout()
    plt.show()

    return matrix


def plot_voltage_profiles(devices,
                          time_col="timestamp_cst",
                          voltage_col="DATA",
                          start_time=None,
                          end_time=None):
    """
    Plot voltage profiles for multiple pre-filtered DataFrames on a single axes.

    devices : list of (df, label) tuples
        df    — DataFrame already filtered to the desired phase(s)
        label — display name for the legend

    Each series is filtered (notna, >0, IQR outlier removal) and
    resampled to 15-min intervals before plotting.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    all_vals = []
    skipped = []
    series_map = {}

    for df, label in devices:
        df = _filter_time(df, time_col, start_time, end_time)
        d = df[df[voltage_col].notna() & (df[voltage_col] > 0)]
        if d.empty:
            skipped.append(label)
            continue
        s = _drop_outliers(d.groupby(time_col)[voltage_col].mean().sort_index())
        if s.empty:
            skipped.append(label)
            continue
        idx = pd.to_datetime(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        s.index = idx
        s = s.resample("15min").mean().dropna()
        if s.empty:
            skipped.append(label)
            continue
        ax.plot(s.index, s.values, linewidth=0.8, label=label, alpha=0.8)
        all_vals.extend(s.values)
        series_map[label] = s

    if skipped:
        print(f"Skipped (no data after filtering): {', '.join(skipped)}")

    if not all_vals:
        plt.close(fig)
        print("No data to plot.")
        return None

    all_arr = np.array(all_vals)
    vmin, vmax = np.nanmin(all_arr), np.nanmax(all_arr)
    margin = (vmax - vmin) * 0.02 or 0.1

    ax.set_xlabel("Timestamp (CST)")
    ax.set_ylabel("Voltage")
    ax.legend(fontsize=8)
    ax.set_ylim(vmin - margin, vmax + margin)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    fig.autofmt_xdate()
    fig.tight_layout()
    ax.set_ylim(vmin - margin, vmax + margin)
    plt.show()

    if len(series_map) < 2:
        return None

    combined = pd.concat(series_map.values(), axis=1, keys=series_map.keys())
    corr = combined.corr()
    print("\nPairwise Pearson correlations:")
    print(corr.to_string())
    return corr


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

    phase_label = f"Phase {phase1}" if phase1 == phase2 else f"Phase {phase1} vs {phase2}"

    if len(s1) < 2:
        print(f"\n[{phase_label}] Insufficient overlapping data to compute correlation.")
        return float("nan")

    corr = _corr(s1, s2)

    # rolling correlation to find negatively correlated windows
    window = min(SLOTS_PER_DAY, len(s1))
    neg_periods = []
    if len(s1) > window:
        with np.errstate(divide="ignore", invalid="ignore"):
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
            seg_corr = _corr(s1.loc[start:end], s2.loc[start:end])
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
                mat.loc[p1, p2] = _corr(a, b)

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
        up_name   = row.get("device_upstream",   "?")
        down_name = row.get("device_downstream", "?")
        up   = f"{up_name} ({up_mslink})"
        down = f"{down_name} ({down_mslink})"

        base = {
            "mslink_upstream":   up_mslink,
            "device_upstream":   row.get("device_upstream"),
            "mslink_downstream": down_mslink,
            "device_downstream": row.get("device_downstream"),
            "feeder":            row.get("feeder"),
        }

        # build 3×3 matrix
        mat = {}
        for p1 in phases:
            for p2 in phases:
                val = row.get(f"{p1}-{p2}", float("nan"))
                mat[(p1, p2)] = float(val) if not pd.isna(val) else float("nan")

        missing_cells = [(p1, p2) for p1 in phases for p2 in phases if pd.isna(mat[(p1, p2)])]
        if missing_cells:
            # whole rows missing → that upstream phase has no data
            miss_up = [p1 for p1 in phases
                       if all(pd.isna(mat[(p1, p2)]) for p2 in phases)]
            # whole columns missing → that downstream phase has no data
            miss_dn = [p2 for p2 in phases
                       if all(pd.isna(mat[(p1, p2)]) for p1 in phases)]
            explained = {(p1, p2) for p1 in miss_up for p2 in phases} | \
                        {(p1, p2) for p2 in miss_dn for p1 in phases}
            individual = [f"{p1}-{p2}" for p1, p2 in missing_cells if (p1, p2) not in explained]
            parts = []
            if miss_up:
                parts.append(f"phase {'/'.join(miss_up)} missing on {up_name}")
            if miss_dn:
                parts.append(f"phase {'/'.join(miss_dn)} missing on {down_name}")
            if individual:
                parts.append(f"missing cells: {', '.join(individual)}")
            reason = "; ".join(parts)
            confusing_rows.append({**base, "reason": reason})
            print(f"[CONFUSING]  {up}  →  {down}  ({reason})")
            continue

        # for each upstream phase, find which downstream phase it correlates best with
        mapping, confusing, reason = {}, False, ""
        for p1 in phases:
            row_vals = {p2: mat[(p1, p2)] for p2 in phases}
            best_p2  = max(row_vals, key=row_vals.get)
            best_val = row_vals[best_p2]
            if best_val < 0.5:
                confusing = True
                reason = f"{up_name} phase {p1} has no strong match (best r={best_val:.3f} to {down_name} phase {best_p2})"
                break
            mapping[p1] = best_p2

        if not confusing:
            claimed = list(mapping.values())
            dups = [p for p in phases if claimed.count(p) > 1]
            if dups:
                confusing = True
                parts = []
                for dup_p2 in dups:
                    claimants = [p1 for p1 in phases if mapping.get(p1) == dup_p2]
                    parts.append(f"{up_name} phases {' & '.join(claimants)} both map to {down_name} phase {dup_p2}")
                reason = "; ".join(parts)

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
                         if (feeder_df[feeder_df["feeder"] == f]["cat"] == "mismatched").sum() == 0
                         and (feeder_df[feeder_df["feeder"] == f]["cat"] == "matching").sum() >= 1]
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
    # write analysis CSV with all pairs, their category, and mapping/reason
    from pathlib import Path
    analysis_path = Path(csv_path).parent / f"{Path(csv_path).stem}__analysis.csv"
    analysis_rows = (
        [{**r, "category": "matching",   "reason": ""} for r in matching_rows]
      + [{**r, "category": "mismatched", "reason": ""} for r in mismatched_rows]
      + [{**r, "category": "confusing",  "mapping": ""} for r in confusing_rows]
    )
    pd.DataFrame(analysis_rows).to_csv(analysis_path, index=False)
    print(f"  Analysis written  →  {analysis_path.name}")
    print(f"────────────────────────────────────────────────────────")

    return {
        "matching":   pd.DataFrame(matching_rows),
        "mismatched": pd.DataFrame(mismatched_rows),
        "confusing":  pd.DataFrame(confusing_rows),
    }


def extract_missing_phase_devices(analysis_csv_path):
    """
    Read the analysis CSV produced by analyze_correlations and extract all
    devices referenced in confusing-pair reason strings as having missing phases.

    Parses reason text of the form "phase X/Y missing on DeviceName" and maps
    the device name back to its mslink using the upstream/downstream columns.

    Returns a DataFrame with columns: mslink, device, feeder, missing_phases.
    """
    import re
    df = pd.read_csv(analysis_csv_path)
    confusing = df[df["category"] == "confusing"].copy()

    # build device_name → (mslink, feeder) lookup from both roles
    name_to_meta = {}
    for _, row in df.iterrows():
        for name_col, mslink_col in [("device_upstream",   "mslink_upstream"),
                                     ("device_downstream", "mslink_downstream")]:
            name   = row.get(name_col)
            mslink = row.get(mslink_col)
            feeder = row.get("feeder")
            if pd.notna(name) and pd.notna(mslink):
                name_to_meta[str(name)] = (mslink, feeder)

    pattern = re.compile(r"phase ([\w/]+) missing on ([^;]+)")

    device_info = {}  # mslink → {device, feeder, missing set}
    for _, row in confusing.iterrows():
        reason = str(row.get("reason", "") or "")
        for match in pattern.finditer(reason):
            phase_str  = match.group(1).strip()
            device_name = match.group(2).strip()
            phases_found = phase_str.split("/")
            meta = name_to_meta.get(device_name)
            if meta is None:
                continue
            mslink, feeder = meta
            entry = device_info.setdefault(mslink,
                        {"device": device_name, "feeder": feeder, "missing": set()})
            entry["missing"].update(phases_found)

    if not device_info:
        print("No devices with missing phases found.")
        return pd.DataFrame(columns=["mslink", "device", "feeder", "missing_phases"])

    records = [
        {"mslink": mslink,
         "device": info["device"],
         "feeder": info["feeder"],
         "missing_phases": ", ".join(sorted(info["missing"]))}
        for mslink, info in device_info.items()
    ]
    result = (pd.DataFrame(records)
                .sort_values(["feeder", "device"])
                .reset_index(drop=True))

    print(f"── Devices with missing phases ({len(result)}) ──────────────────────")
    for _, r in result.iterrows():
        print(f"  {r['device']} ({r['mslink']})  feeder={r['feeder']}  "
              f"missing: {r['missing_phases']}")
    print(f"────────────────────────────────────────────────────────")

    return result


def find_low_correlation_phases(csv_path, threshold=0.6, phases=("A", "B", "C")):
    """
    Read a phase correlation matrix CSV (produced by phase_correlation_matrix)
    and identify every device-phase that has low correlation (< threshold)
    across its entire row or column in at least one pair.

    A device-phase is flagged when ALL of its correlation values with the
    other device's phases are below the threshold:
      - row flag  : upstream_device / phase_X has max(X-A, X-B, X-C) < threshold
      - column flag: downstream_device / phase_Y has max(A-Y, B-Y, C-Y) < threshold

    Returns a DataFrame with columns:
        mslink, device, role (upstream/downstream), phase, max_corr, feeder
    sorted by max_corr ascending.
    """
    df = pd.read_csv(csv_path)
    phase_cols = [f"{p1}-{p2}" for p1 in phases for p2 in phases]
    present = [c for c in phase_cols if c in df.columns]

    records = []

    for _, row in df.iterrows():
        up_mslink  = row.get("mslink_upstream")
        up_device  = row.get("device_upstream")
        dn_mslink  = row.get("mslink_downstream")
        dn_device  = row.get("device_downstream")
        feeder     = row.get("feeder")

        # reconstruct the sub-matrix for this pair
        mat = {}
        for col in present:
            p1, p2 = col.split("-")
            mat.setdefault(p1, {})[p2] = row[col]

        # check each upstream phase (row of the pair matrix)
        for p in phases:
            row_vals = [mat.get(p, {}).get(q, np.nan) for q in phases]
            finite = [v for v in row_vals if not np.isnan(v)]
            if not finite:
                continue
            max_corr = max(finite)
            if max_corr < threshold:
                records.append({
                    "mslink":    up_mslink,
                    "device":    up_device,
                    "role":      "upstream",
                    "phase":     p,
                    "max_corr":  round(max_corr, 4),
                    "feeder":    feeder,
                })

        # check each downstream phase (column of the pair matrix)
        for q in phases:
            col_vals = [mat.get(p, {}).get(q, np.nan) for p in phases]
            finite = [v for v in col_vals if not np.isnan(v)]
            if not finite:
                continue
            max_corr = max(finite)
            if max_corr < threshold:
                records.append({
                    "mslink":    dn_mslink,
                    "device":    dn_device,
                    "role":      "downstream",
                    "phase":     q,
                    "max_corr":  round(max_corr, 4),
                    "feeder":    feeder,
                })

    if not records:
        print(f"No device-phases found with max correlation < {threshold}.")
        return pd.DataFrame(columns=["mslink", "device", "role", "phase", "max_corr", "feeder"])

    result = (pd.DataFrame(records)
                .drop_duplicates()
                .sort_values("max_corr")
                .reset_index(drop=True))

    print(f"── Low-correlation device-phases (threshold={threshold}) "
          f"── {len(result)} found ──")
    for _, r in result.iterrows():
        print(f"  {r['device']} ({r['mslink']})  phase={r['phase']}  "
              f"role={r['role']}  max_corr={r['max_corr']}  feeder={r['feeder']}")
    print("────────────────────────────────────────────────────────")

    return result


def _plot_load_temp_single(df, poi, load_col, temp_col, time_col, color_by, plot=True):
    """Render the two-panel scatter + time-series plot for one feeder."""
    df = df[df[load_col].notna() & df[temp_col].notna()].copy()
    df[load_col] = df[load_col].astype(float)
    df[temp_col] = df[temp_col].astype(float)
    if df.empty:
        print(f"  [{poi}] no data after filtering — skipped")
        return

    x, y = df[temp_col].values, df[load_col].values
    coef = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])

    # per-week stats
    df["_week"] = df[time_col].dt.to_period("W")
    weekly_stats = {}
    for week, wdf in df.groupby("_week"):
        wx, wy = wdf[temp_col].values, wdf[load_col].values
        if len(wx) < 2:
            continue
        wcoef = np.polyfit(wx, wy, 1)
        wr = float(np.corrcoef(wx, wy)[0, 1])
        weekly_stats[str(week)] = (round(wr, 6), round(float(wcoef[0]), 6))

    print(f"  [{poi}] overall  Pearson r={r:.4f}   slope={coef[0]:.4g}   n={len(df):,}")
    for wk, (wr, ws) in weekly_stats.items():
        print(f"    week {wk}  r={wr:.4f}   slope={ws:.4g}")

    if not plot:
        return round(r, 6), round(float(coef[0]), 6), weekly_stats

    if color_by == "hour":
        c_vals, cmap, cb_label = df[time_col].dt.hour, "twilight", "Hour of day"
        vmin, vmax = 0, 23
    elif color_by == "month":
        c_vals, cmap, cb_label = df[time_col].dt.month, "hsv", "Month"
        vmin, vmax = 1, 12
    else:
        c_vals, cmap, cb_label, vmin, vmax = None, None, None, None, None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Feeder: {poi}", fontsize=11)

    # scatter
    ax = axes[0]
    sc_kw = dict(s=8, alpha=0.4, linewidths=0)
    if c_vals is not None:
        sc = ax.scatter(df[temp_col], df[load_col], c=c_vals,
                        cmap=cmap, vmin=vmin, vmax=vmax, **sc_kw)
        fig.colorbar(sc, ax=ax, label=cb_label)
    else:
        ax.scatter(df[temp_col], df[load_col], **sc_kw)

    xline = np.linspace(x.min(), x.max(), 200)

    # per-week lines (faded)
    week_cmap = plt.get_cmap("tab20")
    for idx, (week, wdf) in enumerate(df.groupby("_week")):
        wx, wy = wdf[temp_col].values, wdf[load_col].values
        if len(wx) < 2:
            continue
        wcoef = np.polyfit(wx, wy, 1)
        wr = float(np.corrcoef(wx, wy)[0, 1])
        color = week_cmap(idx % 20)
        ax.plot(xline, np.polyval(wcoef, xline), color=color, linewidth=0.8,
                alpha=0.55, label=f"{week}  r={wr:.2f}")

    # overall line on top
    ax.plot(xline, np.polyval(coef, xline), color="crimson", linewidth=2.0,
            label=f"overall  slope={coef[0]:.3g}  r={r:.3f}", zorder=10)

    ax.set_xlabel(temp_col)
    ax.set_ylabel(load_col)
    ax.set_title("Load vs Temperature")
    ax.legend(fontsize=6, loc="upper left")

    # dual-axis time series
    ax2 = axes[1]
    ts = df.set_index(time_col)[[load_col, temp_col]].sort_index().resample("1h").mean()
    color_load, color_temp = "steelblue", "darkorange"
    ax2.plot(ts.index, ts[load_col], color=color_load, linewidth=0.7,
             label=load_col, alpha=0.85)
    ax2.set_ylabel(load_col, color=color_load)
    ax2.tick_params(axis="y", labelcolor=color_load)
    ax_r = ax2.twinx()
    ax_r.plot(ts.index, ts[temp_col], color=color_temp, linewidth=0.7,
              label=temp_col, alpha=0.85)
    ax_r.set_ylabel(temp_col, color=color_temp)
    ax_r.tick_params(axis="y", labelcolor=color_temp)
    lines = ax2.get_lines() + ax_r.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
    ax2.set_title("Load & Temperature over Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

    return round(r, 6), round(float(coef[0]), 6), weekly_stats


def plot_load_vs_temperature(load_df, temp_df,
                             load_col="load_kw",
                             temp_col="temperature_f",
                             time_col="timestamp",
                             poi_col="poi_id",
                             feeder_col="feeder",
                             color_by="hour",
                             plot=True):
    """
    Join hourly load and temperature DataFrames on (time_col, poi_col) and
    produce one two-panel plot per feeder (feeder_col).

    poi_col identifies the geographic area used as the join key; feeder_col
    is the column in load_df that groups records into feeders.

    Left panel  — scatter of temperature vs load with OLS regression line.
    Right panel — dual-axis time series of load and temperature.

    color_by : "hour"  — colour scatter points by hour of day
               "month" — colour scatter points by month
               None    — no colouring

    Returns the merged DataFrame.
    """
    load_df = load_df.copy()
    temp_df = temp_df.copy()

    load_df[time_col] = pd.to_datetime(load_df[time_col])
    temp_df[time_col] = pd.to_datetime(temp_df[time_col])

    for _df in (load_df, temp_df):
        if _df[time_col].dt.tz is not None:
            _df[time_col] = _df[time_col].dt.tz_convert("UTC").dt.tz_localize(None)

    merged = pd.merge(load_df, temp_df, on=[time_col, poi_col], how="inner",
                      suffixes=("_load", "_temp"))

    if merged.empty:
        print("No overlapping records after join — check column names and poi_id values.")
        return None

    feeders = sorted(merged[feeder_col].dropna().unique())
    print(f"{len(feeders)} feeder(s) found: {feeders}")

    merged["pearson_r"] = float("nan")
    merged["slope"] = float("nan")
    merged["weekly_pearson_r"] = float("nan")
    merged["weekly_slope"] = float("nan")
    merged["_week"] = pd.to_datetime(merged[time_col]).dt.to_period("W").astype(str)

    for feeder in feeders:
        mask = merged[feeder_col] == feeder
        result = _plot_load_temp_single(merged[mask], feeder,
                                        load_col, temp_col, time_col, color_by,
                                        plot=plot)
        if result is None:
            continue
        r, slope, weekly_stats = result
        merged.loc[mask, "pearson_r"] = r
        merged.loc[mask, "slope"] = slope
        for week_str, (wr, ws) in weekly_stats.items():
            week_mask = mask & (merged["_week"] == week_str)
            merged.loc[week_mask, "weekly_pearson_r"] = wr
            merged.loc[week_mask, "weekly_slope"] = ws

    merged.drop(columns=["_week"], inplace=True)
    return merged


def forecast_load_from_temperature(load_df, temp_df,
                                   load_col="load_kw",
                                   temp_col="temperature_f",
                                   time_col="timestamp",
                                   poi_col="poi_id",
                                   feeder_col="feeder",
                                   plot=True,
                                   periodic_r_threshold=0.3):
    """
    Train a linear regression (temperature → load) on week 1 of each feeder,
    forecast week 2 load from week 2 temperatures, and compare against actuals.

    The two calendar weeks are identified automatically from the data; the
    earlier week is used for training and the later for evaluation.

    Returns a dict keyed by feeder with:
        slope, intercept, train_r          — week-1 model
        mae, rmse, forecast_r              — week-2 forecast quality
        week2                              — DataFrame of week-2 rows with
                                             a "load_forecast" column added
    """
    load_df = load_df.copy()
    temp_df = temp_df.copy()

    load_df[time_col] = pd.to_datetime(load_df[time_col])
    temp_df[time_col] = pd.to_datetime(temp_df[time_col])

    for _df in (load_df, temp_df):
        if _df[time_col].dt.tz is not None:
            _df[time_col] = _df[time_col].dt.tz_convert("UTC").dt.tz_localize(None)

    merged = pd.merge(load_df, temp_df, on=[time_col, poi_col], how="inner",
                      suffixes=("_load", "_temp"))

    if merged.empty:
        print("No overlapping records after join.")
        return {}

    merged[load_col] = merged[load_col].astype(float)
    merged[temp_col] = merged[temp_col].astype(float)
    merged = merged[merged[load_col].notna() & merged[temp_col].notna()]

    results = {}
    feeders = sorted(merged[feeder_col].dropna().unique())
    print(f"{len(feeders)} feeder(s): {feeders}")

    for feeder in feeders:
        fdf = merged[merged[feeder_col] == feeder].copy()
        t_start = fdf[time_col].min()
        t_split = t_start + pd.Timedelta(days=7)
        if fdf[time_col].max() <= t_split:
            print(f"  [{feeder}] less than 2 weeks of data — skipped")
            continue
        train = fdf[fdf[time_col] < t_split].copy()
        test  = fdf[fdf[time_col] >= t_split].copy()
        w1 = f"{t_start.date()} – {(t_start + pd.Timedelta(days=6)).date()}"
        w2 = f"{t_split.date()} – {fdf[time_col].max().date()}"
        fdf["_week"] = fdf[time_col].dt.to_period("W")  # for scatter coloring only

        if len(train) < 2 or test.empty:
            print(f"  [{feeder}] insufficient data in one of the weeks — skipped")
            continue

        # shared metric helper
        def _m(ya, yf):
            r = ya - yf
            with np.errstate(divide="ignore", invalid="ignore"):
                return (float(np.mean(np.abs(r))), float(np.sqrt(np.mean(r ** 2))),
                        float(np.corrcoef(ya, yf)[0, 1]))

        # measure temp-load correlation to decide forecast method
        x_train = train[temp_col].values
        y_train = train[load_col].values
        with np.errstate(divide="ignore", invalid="ignore"):
            train_r = float(np.corrcoef(x_train, y_train)[0, 1])
        if np.isnan(train_r):
            train_r = 0.0

        # ── periodic branch ───────────────────────────────────────────────────
        if abs(train_r) < periodic_r_threshold:
            # build (hour, is_weekend) profile from week-1 load
            _tr = train.copy()
            _tr["_hour"] = _tr[time_col].dt.hour
            _tr["_is_wknd"] = (_tr[time_col].dt.dayofweek >= 5)
            profile = (_tr.groupby(["_hour", "_is_wknd"])[load_col]
                       .mean().reset_index().rename(columns={load_col: "_fc"}))
            hour_mean = _tr.groupby("_hour")[load_col].mean()

            _te = test.copy()
            _te["_hour"] = _te[time_col].dt.hour
            _te["_is_wknd"] = (_te[time_col].dt.dayofweek >= 5)
            _te = _te.merge(profile, on=["_hour", "_is_wknd"], how="left")
            miss = _te["_fc"].isna()
            if miss.any():
                _te.loc[miss, "_fc"] = _te.loc[miss, "_hour"].map(hour_mean)
            _te["load_forecast"]      = _te["_fc"]
            _te["load_forecast_quad"] = _te["_fc"]
            _te.drop(columns=["_hour", "_is_wknd", "_fc"], inplace=True)

            y_actual  = _te[load_col].values
            y_fc_per  = _te["load_forecast"].values
            mae, rmse, forecast_r = _m(y_actual, y_fc_per)

            print(f"  [{feeder}]  week 1 ({w1})  temp-load r={train_r:.4f} → periodic forecast")
            print(f"           week 2 ({w2})  MAE={mae:.4g}  RMSE={rmse:.4g}  r={forecast_r:.4f}")

            results[feeder] = {
                "forecast_method": "periodic",
                "slope": float("nan"), "intercept": float("nan"), "train_r": train_r,
                "mae": mae, "rmse": rmse, "forecast_r": forecast_r,
                "mae_quad": mae, "rmse_quad": rmse, "forecast_r_quad": forecast_r,
                "coef_quad": [], "week2": _te,
            }

            if not plot:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"Feeder: {feeder}  [periodic  r(temp,load)={train_r:.3f}]", fontsize=12)

            hours = np.arange(24)

            def _plot_profile(ax_, df_, title_):
                for is_wknd, lbl, clr in [(False, "weekday", "steelblue"),
                                          (True,  "weekend", "darkorange")]:
                    sub = df_[df_[time_col].dt.dayofweek >= 5 if is_wknd
                              else df_[time_col].dt.dayofweek < 5]
                    if sub.empty:
                        continue
                    sub = sub.copy()
                    sub["_h"] = sub[time_col].dt.hour
                    hm = sub.groupby("_h")[load_col].mean()
                    ax_.plot(hours, [hm.get(h, np.nan) for h in hours],
                             "o-", color=clr, linewidth=1.4, markersize=4, label=lbl)
                ax_.set_xlabel("hour of day")
                ax_.set_ylabel(load_col)
                ax_.set_title(title_)
                ax_.legend(fontsize=8)
                ax_.set_xticks(range(0, 24, 3))

            _plot_profile(axes[0, 0], fdf, "Hourly load profile — all data")

            # top-right: dual-axis time series (all data)
            ax2 = axes[0, 1]
            ts_all = fdf.set_index(time_col)[[load_col, temp_col]].sort_index().resample("1h").mean()
            ax2.plot(ts_all.index, ts_all[load_col], color="steelblue",
                     linewidth=0.7, label=load_col, alpha=0.85)
            ax2.set_ylabel(load_col, color="steelblue")
            ax2.tick_params(axis="y", labelcolor="steelblue")
            ax_r2 = ax2.twinx()
            ax_r2.plot(ts_all.index, ts_all[temp_col], color="darkorange",
                       linewidth=0.7, label=temp_col, alpha=0.85)
            ax_r2.set_ylabel(temp_col, color="darkorange")
            ax_r2.tick_params(axis="y", labelcolor="darkorange")
            lines2 = ax2.get_lines() + ax_r2.get_lines()
            ax2.legend(lines2, [l.get_label() for l in lines2], fontsize=8, loc="upper left")
            ax2.set_title("Load & Temperature over Time")

            _plot_profile(axes[1, 0], train, f"Week-1 profile used for forecast  ({w1})")

            # bottom-right: forecast vs actual
            ax4 = axes[1, 1]
            _te_s = _te.set_index(time_col).sort_index().resample("1h").mean()
            ax4.plot(_te_s.index, _te_s[load_col], color="steelblue",
                     linewidth=0.9, label="actual", alpha=0.9)
            ax4.plot(_te_s.index, _te_s["load_forecast"], color="crimson",
                     linewidth=0.9, linestyle="--",
                     label=f"periodic  MAE={mae:.4g}  r={forecast_r:.3f}", alpha=0.9)
            ax4.set_ylabel(load_col)
            ax4r = ax4.twinx()
            ax4r.plot(_te_s.index, _te_s[temp_col], color="seagreen",
                      linewidth=0.7, alpha=0.7, label=temp_col)
            ax4r.set_ylabel(temp_col, color="seagreen")
            ax4r.tick_params(axis="y", labelcolor="seagreen")
            _lines4 = ax4.get_lines() + ax4r.get_lines()
            ax4.legend(_lines4, [l.get_label() for l in _lines4], fontsize=8)
            ax4.set_title(f"Week 2 forecast vs actual  ({w2})")

            fig.autofmt_xdate()
            fig.tight_layout()
            plt.show()
            continue

        # ── temperature-based branch ──────────────────────────────────────────
        coef1 = np.polyfit(x_train, y_train, 1)
        coef2 = np.polyfit(x_train, y_train, 2)
        slope, intercept = float(coef1[0]), float(coef1[1])

        # forecast week 2 with both models
        x_test = test[temp_col].values
        y_actual = test[load_col].values
        y_fc_lin  = np.polyval(coef1, x_test)
        y_fc_quad = np.polyval(coef2, x_test)
        test = test.copy()
        test["load_forecast"]      = y_fc_lin
        test["load_forecast_quad"] = y_fc_quad

        mae,  rmse,  forecast_r  = _m(y_actual, y_fc_lin)
        mae2, rmse2, forecast_r2 = _m(y_actual, y_fc_quad)

        print(f"  [{feeder}]  week 1 ({w1})  r={train_r:.4f}  "
              f"slope={slope:.4g}  intercept={intercept:.4g}")
        print(f"           week 2 ({w2})  linear:    MAE={mae:.4g}  RMSE={rmse:.4g}  r={forecast_r:.4f}")
        print(f"                          quadratic: MAE={mae2:.4g}  RMSE={rmse2:.4g}  r={forecast_r2:.4f}")

        results[feeder] = {
            "forecast_method": "temperature",
            "slope": slope, "intercept": intercept, "train_r": train_r,
            "mae": mae, "rmse": rmse, "forecast_r": forecast_r,
            "mae_quad": mae2, "rmse_quad": rmse2, "forecast_r_quad": forecast_r2,
            "coef_quad": coef2.tolist(), "week2": test,
        }

        if not plot:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Feeder: {feeder}", fontsize=12)

        all_x = fdf[temp_col].values
        all_y = fdf[load_col].values
        overall_coef = np.polyfit(all_x, all_y, 1)
        overall_r    = float(np.corrcoef(all_x, all_y)[0, 1])
        xline_all    = np.linspace(all_x.min(), all_x.max(), 200)

        # ── top-left: temp vs load scatter (all data) with per-week lines ──
        ax = axes[0, 0]
        ax.scatter(all_x, all_y, s=6, alpha=0.3, linewidths=0, color="steelblue")
        week_cmap = plt.get_cmap("tab20")
        for idx, (week, wdf_) in enumerate(fdf.groupby("_week")):
            wx, wy = wdf_[temp_col].values, wdf_[load_col].values
            if len(wx) < 2:
                continue
            wcoef = np.polyfit(wx, wy, 1)
            wr    = float(np.corrcoef(wx, wy)[0, 1])
            ax.plot(xline_all, np.polyval(wcoef, xline_all),
                    color=week_cmap(idx % 20), linewidth=0.9, alpha=0.6,
                    label=f"{week}  r={wr:.2f}")
        ax.plot(xline_all, np.polyval(overall_coef, xline_all),
                color="crimson", linewidth=2.0, zorder=10,
                label=f"overall  r={overall_r:.3f}")
        ax.set_xlabel(temp_col)
        ax.set_ylabel(load_col)
        ax.set_title("Load vs Temperature (all data)")
        ax.legend(fontsize=6, loc="upper left")

        # ── top-right: dual-axis time series (all data) ──────────────────────
        ax2 = axes[0, 1]
        ts_all = fdf.set_index(time_col)[[load_col, temp_col]].sort_index().resample("1h").mean()
        color_load, color_temp = "steelblue", "darkorange"
        ax2.plot(ts_all.index, ts_all[load_col], color=color_load,
                 linewidth=0.7, label=load_col, alpha=0.85)
        ax2.set_ylabel(load_col, color=color_load)
        ax2.tick_params(axis="y", labelcolor=color_load)
        ax_r2 = ax2.twinx()
        ax_r2.plot(ts_all.index, ts_all[temp_col], color=color_temp,
                   linewidth=0.7, label=temp_col, alpha=0.85)
        ax_r2.set_ylabel(temp_col, color=color_temp)
        ax_r2.tick_params(axis="y", labelcolor=color_temp)
        lines2 = ax2.get_lines() + ax_r2.get_lines()
        ax2.legend(lines2, [l.get_label() for l in lines2], fontsize=8, loc="upper left")
        ax2.set_title("Load & Temperature over Time")

        # ── bottom-left: week 1 scatter + fit line ────────────────────────────
        ax3 = axes[1, 0]
        ax3.scatter(x_train, y_train, s=8, alpha=0.45, linewidths=0,
                    label=f"week 1 data  (r={train_r:.3f})")
        xline_w1 = np.linspace(x_train.min(), x_train.max(), 200)
        ax3.plot(xline_w1, np.polyval(coef1, xline_w1), color="crimson",
                 linewidth=1.5, label=f"linear  slope={slope:.3g}")
        ax3.plot(xline_w1, np.polyval(coef2, xline_w1), color="darkorange",
                 linewidth=1.5, linestyle="--", label="quadratic")
        ax3.set_xlabel(temp_col)
        ax3.set_ylabel(load_col)
        ax3.set_title(f"Week 1 training fit  ({w1})")
        ax3.legend(fontsize=8)

        # ── bottom-right: week 2 actual vs forecast ───────────────────────────
        ax4 = axes[1, 1]
        test_sorted = test.set_index(time_col).sort_index().resample("1h").mean()
        ax4.plot(test_sorted.index, test_sorted[load_col], color="steelblue",
                 linewidth=0.9, label="actual", alpha=0.9)
        ax4.plot(test_sorted.index, test_sorted["load_forecast"], color="crimson",
                 linewidth=0.9, linestyle="--",
                 label=f"linear  MAE={mae:.4g}  r={forecast_r:.3f}", alpha=0.9)
        ax4.plot(test_sorted.index, test_sorted["load_forecast_quad"], color="darkorange",
                 linewidth=0.9, linestyle=":",
                 label=f"quadratic  MAE={mae2:.4g}  r={forecast_r2:.3f}", alpha=0.9)
        ax4.set_ylabel(load_col)
        ax4r = ax4.twinx()
        ax4r.plot(test_sorted.index, test_sorted[temp_col], color="seagreen",
                  linewidth=0.7, alpha=0.7, label=temp_col)
        ax4r.set_ylabel(temp_col, color="seagreen")
        ax4r.tick_params(axis="y", labelcolor="seagreen")
        _lines4 = ax4.get_lines() + ax4r.get_lines()
        ax4.legend(_lines4, [l.get_label() for l in _lines4], fontsize=8)
        ax4.set_title(f"Week 2 forecast vs actual  ({w2})")

        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()

    return results


def compare_daily_peaks(load_df, temp_df,
                        load_col="load_kw",
                        temp_col="temperature_f",
                        time_col="timestamp",
                        poi_col="poi_id",
                        feeder_col="feeder",
                        plot=True):
    """
    For each feeder and each calendar day, find the hour of peak load and
    the hour of peak temperature, then compute the offset between them.

    offset_hours = peak_load_hour - peak_temp_hour
      positive → load peaks AFTER temperature (load lags temperature)
      negative → load peaks BEFORE temperature (load leads temperature)

    When plot=True, renders a two-panel scatter per feeder:
      left  — load vs temperature, original alignment
      right — load vs temperature after shifting load by -median_offset hours
               so peaks are as closely aligned as possible

    Returns a tuple (detail_df, summary_df):
        detail_df  — one row per (feeder, date) with peak hours and offset
        summary_df — one row per feeder with mean/median/std offset,
                     direction, and the applied shift in the aligned plot
    """
    load_df = load_df.copy()
    temp_df = temp_df.copy()

    load_df[time_col] = pd.to_datetime(load_df[time_col])
    temp_df[time_col] = pd.to_datetime(temp_df[time_col])

    for _df in (load_df, temp_df):
        if _df[time_col].dt.tz is not None:
            _df[time_col] = _df[time_col].dt.tz_convert("UTC").dt.tz_localize(None)

    merged = pd.merge(load_df, temp_df, on=[time_col, poi_col], how="inner",
                      suffixes=("_load", "_temp"))

    if merged.empty:
        print("No overlapping records after join.")
        return pd.DataFrame(), pd.DataFrame()

    merged[load_col] = merged[load_col].astype(float)
    merged[temp_col] = merged[temp_col].astype(float)
    merged = merged[merged[load_col].notna() & merged[temp_col].notna()]
    merged["_date"] = merged[time_col].dt.date
    merged["_hour"] = merged[time_col].dt.hour

    detail_rows = []
    for (feeder, date), day in merged.groupby([feeder_col, "_date"]):
        if day.empty:
            continue
        peak_load_row = day.loc[day[load_col].idxmax()]
        peak_temp_row = day.loc[day[temp_col].idxmax()]
        peak_load_hour = int(peak_load_row["_hour"])
        peak_temp_hour = int(peak_temp_row["_hour"])
        offset = peak_load_hour - peak_temp_hour
        detail_rows.append({
            "feeder":          feeder,
            "date":            date,
            "peak_load_hour":  peak_load_hour,
            "peak_load_value": round(float(peak_load_row[load_col]), 4),
            "peak_temp_hour":  peak_temp_hour,
            "peak_temp_value": round(float(peak_temp_row[temp_col]), 4),
            "offset_hours":    offset,
        })

    if not detail_rows:
        return pd.DataFrame(), pd.DataFrame()

    detail_df = pd.DataFrame(detail_rows).sort_values(["feeder", "date"]).reset_index(drop=True)

    summary_rows = []
    for feeder, grp in detail_df.groupby("feeder"):
        offsets    = grp["offset_hours"]
        mean_off   = offsets.mean()
        median_off = offsets.median()
        std_off    = offsets.std()
        shift      = -int(round(median_off))   # hours to shift load to align peaks
        direction  = ("load lags temperature" if mean_off > 0
                      else "load leads temperature" if mean_off < 0
                      else "no offset")
        summary_rows.append({
            "feeder":          feeder,
            "mean_offset_h":   round(mean_off,   3),
            "median_offset_h": round(median_off, 3),
            "std_offset_h":    round(std_off,    3),
            "direction":       direction,
            "applied_shift_h": shift,
            "n_days":          len(grp),
        })
        print(f"  [{feeder}]  mean offset={mean_off:+.1f}h  "
              f"median={median_off:+.1f}h  std={std_off:.1f}h  "
              f"→ {direction}  (applied shift: {shift:+d}h)")

        if not plot:
            continue

        # ── build original and shifted merged data for this feeder ───────────
        fload = load_df[load_df[feeder_col] == feeder].copy()
        fmerge_orig = pd.merge(fload, temp_df, on=[time_col, poi_col], how="inner",
                               suffixes=("_load", "_temp"))
        fmerge_orig[load_col] = fmerge_orig[load_col].astype(float)
        fmerge_orig[temp_col] = fmerge_orig[temp_col].astype(float)
        fmerge_orig = fmerge_orig[fmerge_orig[load_col].notna() & fmerge_orig[temp_col].notna()]

        # shift load timestamps by `shift` hours, then re-merge with temp
        fload_shifted = fload.copy()
        fload_shifted[time_col] = fload_shifted[time_col] + pd.Timedelta(hours=shift)
        fmerge_shift = pd.merge(fload_shifted, temp_df, on=[time_col, poi_col], how="inner",
                                suffixes=("_load", "_temp"))
        fmerge_shift[load_col] = fmerge_shift[load_col].astype(float)
        fmerge_shift[temp_col] = fmerge_shift[temp_col].astype(float)
        fmerge_shift = fmerge_shift[fmerge_shift[load_col].notna() & fmerge_shift[temp_col].notna()]

        def _draw_row(axes_row, df_, row_label):
            """Fill one row (scatter + dual-axis time series) for a merged df."""
            x = df_[temp_col].values
            y = df_[load_col].values

            # left: scatter with regression line
            ax_s = axes_row[0]
            ax_s.scatter(x, y, s=7, alpha=0.35, linewidths=0, color="steelblue")
            if len(x) >= 2:
                coef  = np.polyfit(x, y, 1)
                r     = float(np.corrcoef(x, y)[0, 1])
                xline = np.linspace(x.min(), x.max(), 200)
                ax_s.plot(xline, np.polyval(coef, xline), color="crimson",
                          linewidth=1.5, label=f"slope={coef[0]:.3g}  r={r:.3f}")
                ax_s.legend(fontsize=8)
            ax_s.set_xlabel(temp_col)
            ax_s.set_ylabel(load_col)
            ax_s.set_title(f"{row_label} — Load vs Temperature")

            # right: dual-axis time series
            ax_t = axes_row[1]
            ts = df_.set_index(time_col)[[load_col, temp_col]].sort_index().resample("1h").mean()
            ax_t.plot(ts.index, ts[load_col], color="steelblue", linewidth=0.7,
                      label=load_col, alpha=0.85)
            ax_t.set_ylabel(load_col, color="steelblue")
            ax_t.tick_params(axis="y", labelcolor="steelblue")
            ax_tr = ax_t.twinx()
            ax_tr.plot(ts.index, ts[temp_col], color="darkorange", linewidth=0.7,
                       label=temp_col, alpha=0.85)
            ax_tr.set_ylabel(temp_col, color="darkorange")
            ax_tr.tick_params(axis="y", labelcolor="darkorange")
            lines_ = ax_t.get_lines() + ax_tr.get_lines()
            ax_t.legend(lines_, [l.get_label() for l in lines_], fontsize=8, loc="upper left")
            ax_t.set_title(f"{row_label} — Load & Temperature over Time")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Feeder: {feeder}  —  peak offset analysis", fontsize=12)
        _draw_row(axes[0], fmerge_orig,  f"Original (median offset={median_off:+.1f}h)")
        _draw_row(axes[1], fmerge_shift, f"Shifted load {shift:+d}h")
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()

    summary_df = pd.DataFrame(summary_rows).reset_index(drop=True)
    return detail_df, summary_df


def forecast_load_with_offset(load_df, temp_df,
                              load_col="load_kw",
                              temp_col="temperature_f",
                              time_col="timestamp",
                              poi_col="poi_id",
                              feeder_col="feeder",
                              plot=True,
                              periodic_r_threshold=0.3):
    """
    Offset-aware two-week load forecast per feeder.

    1. Compute the median daily peak offset for week 1
           offset = median(peak_load_hour - peak_temp_hour)
    2. Shift week-1 load timestamps by -offset hours so load and temperature
       peaks are aligned, then fit a linear regression (temp → shifted load).
    3. Apply the model to week-2 temperatures.  The resulting forecast is in
       "shifted time", so shift forecast timestamps forward by +offset hours
       to put them back in original time coordinates.
    4. Merge the re-shifted forecast with actual week-2 load on timestamp and
       compute MAE, RMSE, and Pearson r.

    Returns a dict keyed by feeder containing:
        offset_h            — median peak offset used
        slope, intercept    — week-1 model (fitted on shifted data)
        train_r             — Pearson r of fit on shifted week-1 data
        mae, rmse, forecast_r — week-2 evaluation metrics
        week2               — DataFrame with actual load and load_forecast
    """
    load_df = load_df.copy()
    temp_df = temp_df.copy()

    load_df[time_col] = pd.to_datetime(load_df[time_col])
    temp_df[time_col] = pd.to_datetime(temp_df[time_col])

    for _df in (load_df, temp_df):
        if _df[time_col].dt.tz is not None:
            _df[time_col] = _df[time_col].dt.tz_convert("UTC").dt.tz_localize(None)

    merged = pd.merge(load_df, temp_df, on=[time_col, poi_col], how="inner",
                      suffixes=("_load", "_temp"))

    if merged.empty:
        print("No overlapping records after join.")
        return {}

    merged[load_col] = merged[load_col].astype(float)
    merged[temp_col] = merged[temp_col].astype(float)
    merged = merged[merged[load_col].notna() & merged[temp_col].notna()]
    results = {}
    feeders = sorted(merged[feeder_col].dropna().unique())
    print(f"{len(feeders)} feeder(s): {feeders}")

    for feeder in feeders:
        fdf = merged[merged[feeder_col] == feeder].copy()
        t_start = fdf[time_col].min()
        t_split = t_start + pd.Timedelta(days=7)
        if fdf[time_col].max() <= t_split:
            print(f"  [{feeder}] less than 2 weeks of data — skipped")
            continue
        train_orig = fdf[fdf[time_col] < t_split].copy()
        test_orig  = fdf[fdf[time_col] >= t_split].copy()
        w1 = f"{t_start.date()} – {(t_start + pd.Timedelta(days=6)).date()}"
        w2 = f"{t_split.date()} – {fdf[time_col].max().date()}"

        if len(train_orig) < 2 or test_orig.empty:
            print(f"  [{feeder}] insufficient data in one of the weeks — skipped")
            continue

        # shared metric helper
        def _m(ya, yf):
            r = ya - yf
            with np.errstate(divide="ignore", invalid="ignore"):
                return (float(np.mean(np.abs(r))), float(np.sqrt(np.mean(r ** 2))),
                        float(np.corrcoef(ya, yf)[0, 1]))

        # measure temp-load correlation on week-1 merged data to decide method
        with np.errstate(divide="ignore", invalid="ignore"):
            _x = train_orig[temp_col].values
            _y = train_orig[load_col].values
            train_r_check = float(np.corrcoef(_x, _y)[0, 1]) if len(_x) >= 2 else 0.0
        if np.isnan(train_r_check):
            train_r_check = 0.0

        import matplotlib.gridspec as gridspec
        import matplotlib.dates as mdates

        # ── periodic branch ───────────────────────────────────────────────────
        if abs(train_r_check) < periodic_r_threshold:
            train_r = train_r_check

            # build profile from raw week-1 load (no temp join needed)
            fload_w1p = load_df[
                (load_df[feeder_col] == feeder) &
                (load_df[time_col] >= t_start) &
                (load_df[time_col] < t_split)
            ].copy()
            fload_w1p[load_col] = fload_w1p[load_col].astype(float)
            fload_w1p["_hour"]   = fload_w1p[time_col].dt.hour
            fload_w1p["_is_wknd"] = (fload_w1p[time_col].dt.dayofweek >= 5)
            profile = (fload_w1p.groupby(["_hour", "_is_wknd"])[load_col]
                       .mean().reset_index().rename(columns={load_col: "_fc"}))
            hour_mean = fload_w1p.groupby("_hour")[load_col].mean()

            fload_w2p = load_df[
                (load_df[feeder_col] == feeder) &
                (load_df[time_col] >= t_split)
            ].copy()
            fload_w2p[load_col] = fload_w2p[load_col].astype(float)
            fload_w2p["_hour"]   = fload_w2p[time_col].dt.hour
            fload_w2p["_is_wknd"] = (fload_w2p[time_col].dt.dayofweek >= 5)
            fload_w2p = fload_w2p.merge(profile, on=["_hour", "_is_wknd"], how="left")
            miss = fload_w2p["_fc"].isna()
            if miss.any():
                fload_w2p.loc[miss, "_fc"] = fload_w2p.loc[miss, "_hour"].map(hour_mean)
            fload_w2p["load_forecast"]      = fload_w2p["_fc"]
            fload_w2p["load_forecast_quad"] = fload_w2p["_fc"]
            fload_w2p.drop(columns=["_hour", "_is_wknd", "_fc"], inplace=True)

            week2 = (fload_w2p.set_index(time_col)[[load_col, "load_forecast", "load_forecast_quad"]]
                     .resample("1h").mean()
                     .dropna()
                     .reset_index())

            if week2.empty:
                print(f"  [{feeder}] no week-2 load data — skipped")
                continue

            y_actual = week2[load_col].values
            y_fc_per = week2["load_forecast"].values
            mae, rmse, forecast_r = _m(y_actual, y_fc_per)

            print(f"  [{feeder}]  week 1 ({w1})  temp-load r={train_r:.4f} → periodic forecast")
            print(f"           week 2 ({w2})  MAE={mae:.4g}  RMSE={rmse:.4g}  r={forecast_r:.4f}")

            results[feeder] = {
                "forecast_method": "periodic",
                "offset_h": 0, "slope": float("nan"), "intercept": float("nan"),
                "train_r": train_r, "mae": mae, "rmse": rmse, "forecast_r": forecast_r,
                "mae_quad": mae, "rmse_quad": rmse, "forecast_r_quad": forecast_r,
                "coef_quad": [], "week2": week2,
            }

            if not plot:
                continue

            fig = plt.figure(figsize=(14, 10))
            fig.suptitle(f"Feeder: {feeder}  [periodic  r(temp,load)={train_r:.3f}]", fontsize=12)
            gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
            ax_prof = fig.add_subplot(gs[0, 0])
            ax_ts1  = fig.add_subplot(gs[0, 1])
            ax_fc   = fig.add_subplot(gs[1, :])

            # top-left: week-1 hourly profile (weekday vs weekend)
            hours = np.arange(24)
            for is_wknd, lbl, clr in [(False, "weekday", "steelblue"),
                                      (True,  "weekend", "darkorange")]:
                sub = fload_w1p[fload_w1p[time_col].dt.dayofweek >= 5
                                if is_wknd else fload_w1p[time_col].dt.dayofweek < 5]
                if sub.empty:
                    continue
                sub = sub.copy()
                sub["_h"] = sub[time_col].dt.hour
                hm = sub.groupby("_h")[load_col].mean()
                ax_prof.plot(hours, [hm.get(h, np.nan) for h in hours],
                             "o-", color=clr, linewidth=1.4, markersize=4, label=lbl)
            ax_prof.set_xlabel("hour of day")
            ax_prof.set_ylabel(load_col)
            ax_prof.set_title(f"Week-1 hourly load profile  ({w1})")
            ax_prof.legend(fontsize=8)
            ax_prof.set_xticks(range(0, 24, 3))

            # top-right: week-1 load time series only
            ts_load_w1 = (fload_w1p.set_index(time_col)[load_col]
                          .groupby(level=0).mean()
                          .resample("1h").mean())
            ax_ts1.plot(ts_load_w1.index, ts_load_w1.values, color="steelblue",
                        linewidth=0.7, label=load_col, alpha=0.85)
            ax_ts1.set_ylabel(load_col)
            ax_ts1.set_title(f"Week 1 — Load  ({w1})")
            ax_ts1.legend(fontsize=8)
            ax_ts1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            plt.setp(ax_ts1.get_xticklabels(), rotation=30, ha="right")

            # bottom: week-2 actual vs periodic forecast
            w2s = (week2.set_index(time_col)[[load_col, "load_forecast"]]
                   .resample("1h").mean())
            _ts_temp_fc = (temp_df[temp_df[time_col] >= t_split]
                           .set_index(time_col)[temp_col].astype(float)
                           .groupby(level=0).mean().resample("1h").mean())
            ax_fc.plot(w2s.index, w2s[load_col], color="steelblue", linewidth=0.9,
                       label="actual", alpha=0.9)
            ax_fc.plot(w2s.index, w2s["load_forecast"], color="crimson", linewidth=0.9,
                       linestyle="--",
                       label=f"periodic  MAE={mae:.4g}  r={forecast_r:.3f}", alpha=0.9)
            ax_fc.set_ylabel(load_col)
            ax_fcr = ax_fc.twinx()
            ax_fcr.plot(_ts_temp_fc.index, _ts_temp_fc.values, color="seagreen",
                        linewidth=0.7, alpha=0.7, label=temp_col)
            ax_fcr.set_ylabel(temp_col, color="seagreen")
            ax_fcr.tick_params(axis="y", labelcolor="seagreen")
            _lines_fc = ax_fc.get_lines() + ax_fcr.get_lines()
            ax_fc.legend(_lines_fc, [l.get_label() for l in _lines_fc], fontsize=8)
            ax_fc.set_title(f"Week 2 — forecast vs actual  ({w2})")
            ax_fc.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            plt.setp(ax_fc.get_xticklabels(), rotation=30, ha="right")

            fig.tight_layout()
            plt.show()
            continue

        # ── temperature-based branch ──────────────────────────────────────────
        # ── step 1: compute median peak offset on week 1 ─────────────────────
        train_orig["_date"] = train_orig[time_col].dt.date
        train_orig["_hour"] = train_orig[time_col].dt.hour
        day_offsets = []
        for _, day in train_orig.groupby("_date"):
            if day.empty:
                continue
            pl = int(day.loc[day[load_col].idxmax(), "_hour"])
            pt = int(day.loc[day[temp_col].idxmax(), "_hour"])
            day_offsets.append(pl - pt)
        if not day_offsets:
            print(f"  [{feeder}] could not compute peak offset — skipped")
            continue
        offset_h = int(round(float(np.median(day_offsets))))

        # ── step 2: shift week-1 load back, re-merge, fit model ──────────────
        fload_w1 = load_df[
            (load_df[feeder_col] == feeder) &
            (load_df[time_col] >= t_start) &
            (load_df[time_col] < t_split)
        ].copy()
        fload_w1_shifted = fload_w1.copy()
        fload_w1_shifted[time_col] = fload_w1_shifted[time_col] - pd.Timedelta(hours=offset_h)

        train_shifted = pd.merge(fload_w1_shifted, temp_df, on=[time_col, poi_col],
                                 how="inner", suffixes=("_load", "_temp"))
        train_shifted[load_col] = train_shifted[load_col].astype(float)
        train_shifted[temp_col] = train_shifted[temp_col].astype(float)
        train_shifted = train_shifted[
            train_shifted[load_col].notna() & train_shifted[temp_col].notna()
        ]

        if len(train_shifted) < 2:
            print(f"  [{feeder}] insufficient overlap after shifting week-1 — skipped")
            continue

        # aggregate to one value per hour so the scatter has clean unique points
        train_agg = (train_shifted
                     .set_index(time_col)[[load_col, temp_col]]
                     .resample("1h").mean()
                     .dropna()
                     .reset_index())

        x_train = train_agg[temp_col].values
        y_train = train_agg[load_col].values
        coef1 = np.polyfit(x_train, y_train, 1)
        coef2 = np.polyfit(x_train, y_train, 2)
        slope, intercept = float(coef1[0]), float(coef1[1])
        train_r = float(np.corrcoef(x_train, y_train)[0, 1])

        # ── step 3: forecast week 2, re-shift timestamps forward ─────────────
        ftemp_w2 = temp_df[
            temp_df[time_col] >= t_split
        ].copy()
        ftemp_w2[temp_col] = ftemp_w2[temp_col].astype(float)

        # forecast is in shifted time; move timestamps +offset_h to restore
        ftemp_w2["load_forecast"]      = np.polyval(coef1, ftemp_w2[temp_col].values)
        ftemp_w2["load_forecast_quad"] = np.polyval(coef2, ftemp_w2[temp_col].values)
        ftemp_w2[time_col] = ftemp_w2[time_col] + pd.Timedelta(hours=offset_h)

        # ── step 4: merge forecast with actual week-2 load, compute metrics ──
        fload_w2 = load_df[
            (load_df[feeder_col] == feeder) &
            (load_df[time_col] >= t_split)
        ].copy()
        fload_w2[load_col] = fload_w2[load_col].astype(float)

        week2_raw = pd.merge(
            fload_w2[[time_col, feeder_col, load_col]],
            ftemp_w2[[time_col, "load_forecast", "load_forecast_quad"]],
            on=time_col, how="inner"
        ).dropna(subset=[load_col, "load_forecast", "load_forecast_quad"])

        if week2_raw.empty:
            print(f"  [{feeder}] no overlapping records for week-2 comparison — skipped")
            continue

        # aggregate to hourly mean to eliminate duplicates from multiple poi_ids
        week2 = (week2_raw
                 .set_index(time_col)[[load_col, "load_forecast", "load_forecast_quad"]]
                 .resample("1h").mean()
                 .dropna()
                 .reset_index())

        y_actual  = week2[load_col].values
        y_fc_lin  = week2["load_forecast"].values
        y_fc_quad = week2["load_forecast_quad"].values
        res_lin   = y_actual - y_fc_lin
        res_quad  = y_actual - y_fc_quad
        mae       = float(np.mean(np.abs(res_lin)))
        rmse      = float(np.sqrt(np.mean(res_lin ** 2)))
        mae2      = float(np.mean(np.abs(res_quad)))
        rmse2     = float(np.sqrt(np.mean(res_quad ** 2)))
        with np.errstate(divide="ignore", invalid="ignore"):
            forecast_r  = float(np.corrcoef(y_actual, y_fc_lin)[0, 1])
            forecast_r2 = float(np.corrcoef(y_actual, y_fc_quad)[0, 1])

        print(f"  [{feeder}]  offset={offset_h:+d}h  "
              f"week-1 r={train_r:.4f}  slope={slope:.4g}")
        print(f"           week-2  linear:    MAE={mae:.4g}  RMSE={rmse:.4g}  r={forecast_r:.4f}")
        print(f"                   quadratic: MAE={mae2:.4g}  RMSE={rmse2:.4g}  r={forecast_r2:.4f}")

        results[feeder] = {
            "forecast_method": "temperature",
            "offset_h": offset_h, "slope": slope, "intercept": intercept,
            "train_r": train_r, "mae": mae, "rmse": rmse, "forecast_r": forecast_r,
            "mae_quad": mae2, "rmse_quad": rmse2, "forecast_r_quad": forecast_r2,
            "coef_quad": coef2.tolist(), "week2": week2,
        }

        if not plot:
            continue

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"Feeder: {feeder}  (offset={offset_h:+d}h)", fontsize=12)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax_scatter = fig.add_subplot(gs[0, 0])
        ax_ts1     = fig.add_subplot(gs[0, 1])
        ax_fc      = fig.add_subplot(gs[1, :])   # full-width bottom

        # top-left: shifted week-1 scatter + fit
        ax_scatter.scatter(x_train, y_train, s=8, alpha=0.4, linewidths=0)
        xline = np.linspace(x_train.min(), x_train.max(), 200)
        ax_scatter.plot(xline, np.polyval(coef1, xline), color="crimson", linewidth=1.5,
                        label=f"linear  slope={slope:.3g}  r={train_r:.3f}")
        ax_scatter.plot(xline, np.polyval(coef2, xline), color="darkorange", linewidth=1.5,
                        linestyle="--", label="quadratic")
        ax_scatter.set_xlabel(temp_col)
        ax_scatter.set_ylabel(load_col)
        ax_scatter.set_title(f"Week 1 (shifted {-offset_h:+d}h) — training fit  ({w1})")
        ax_scatter.legend(fontsize=8)

        # top-right: week-1 dual-axis time series built from the raw source
        # DataFrames so coverage isn't limited to the inner-join overlap
        _load_w1 = load_df[
            (load_df[feeder_col] == feeder) &
            (load_df[time_col] >= t_start) &
            (load_df[time_col] < t_split)
        ].copy()
        _temp_w1 = temp_df[
            (temp_df[time_col] >= t_start) & (temp_df[time_col] < t_split)
        ].copy()
        _load_w1[load_col] = _load_w1[load_col].astype(float)
        _temp_w1[temp_col] = _temp_w1[temp_col].astype(float)

        ts_load = (_load_w1.set_index(time_col)[load_col]
                   .groupby(level=0).mean()
                   .resample("1h").mean())   # keep NaN gaps, don't dropna
        ts_temp = (_temp_w1.set_index(time_col)[temp_col]
                   .groupby(level=0).mean()
                   .resample("1h").mean())

        ax_ts1.plot(ts_load.index, ts_load.values, color="steelblue",
                    linewidth=0.7, label=load_col, alpha=0.85)
        ax_ts1.set_ylabel(load_col, color="steelblue")
        ax_ts1.tick_params(axis="y", labelcolor="steelblue")
        ax_ts1r = ax_ts1.twinx()
        ax_ts1r.plot(ts_temp.index, ts_temp.values, color="darkorange",
                     linewidth=0.7, label=temp_col, alpha=0.85)
        ax_ts1r.set_ylabel(temp_col, color="darkorange")
        ax_ts1r.tick_params(axis="y", labelcolor="darkorange")
        lines1 = ax_ts1.get_lines() + ax_ts1r.get_lines()
        ax_ts1.legend(lines1, [l.get_label() for l in lines1], fontsize=8, loc="upper left")
        ax_ts1.set_title(f"Week 1 — Load & Temperature  ({w1})")
        ax_ts1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.setp(ax_ts1.get_xticklabels(), rotation=30, ha="right")

        # bottom: week-2 actual vs forecast (full width)
        # resample without dropna so NaN gaps break the line instead of
        # connecting across missing timestamps with vertical-looking lines
        w2s = (week2.set_index(time_col)[[load_col, "load_forecast", "load_forecast_quad"]]
               .resample("1h").mean())
        _ts_temp_fc = (temp_df[temp_df[time_col] >= t_split]
                       .set_index(time_col)[temp_col].astype(float)
                       .groupby(level=0).mean().resample("1h").mean())
        ax_fc.plot(w2s.index, w2s[load_col], color="steelblue", linewidth=0.9,
                   label="actual", alpha=0.9)
        ax_fc.plot(w2s.index, w2s["load_forecast"], color="crimson", linewidth=0.9,
                   linestyle="--",
                   label=f"linear  MAE={mae:.4g}  r={forecast_r:.3f}", alpha=0.9)
        ax_fc.plot(w2s.index, w2s["load_forecast_quad"], color="darkorange", linewidth=0.9,
                   linestyle=":",
                   label=f"quadratic  MAE={mae2:.4g}  r={forecast_r2:.3f}", alpha=0.9)
        ax_fc.set_ylabel(load_col)
        ax_fcr = ax_fc.twinx()
        ax_fcr.plot(_ts_temp_fc.index, _ts_temp_fc.values, color="seagreen",
                    linewidth=0.7, alpha=0.7, label=temp_col)
        ax_fcr.set_ylabel(temp_col, color="seagreen")
        ax_fcr.tick_params(axis="y", labelcolor="seagreen")
        _lines_fc = ax_fc.get_lines() + ax_fcr.get_lines()
        ax_fc.legend(_lines_fc, [l.get_label() for l in _lines_fc], fontsize=8)
        ax_fc.set_title(f"Week 2 — forecast vs actual  ({w2})")
        ax_fc.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        plt.setp(ax_fc.get_xticklabels(), rotation=30, ha="right")

        fig.tight_layout()
        plt.show()

    return results


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
