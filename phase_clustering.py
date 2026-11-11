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
                        end_time=None,
                        cross_corr=False,
                        max_lag_hours=24):
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

    # --- cross-correlation: find the lag that maximises Pearson r ----------
    best_lag = 0
    best_lag_corr = corr
    lag_corr_series = None
    if cross_corr and len(s1) >= 4:
        # determine slot size from the index
        diffs = pd.Series(s1.index).diff().dropna()
        slot = diffs.median() if not diffs.empty else pd.Timedelta(minutes=15)
        max_slots = max(1, int(pd.Timedelta(hours=max_lag_hours) / slot))
        max_slots = min(max_slots, len(s1) - 2)

        lags = range(-max_slots, max_slots + 1)
        lag_corrs = []
        for lag in lags:
            if lag == 0:
                lag_corrs.append(corr)
            elif lag > 0:
                r = _corr(s1.iloc[lag:], s2.iloc[:-lag])
            else:
                r = _corr(s1.iloc[:lag], s2.iloc[-lag:])
            if lag != 0:
                lag_corrs.append(r)

        lag_corr_series = pd.Series(lag_corrs, index=list(lags))
        best_idx = int(np.nanargmax(np.abs(lag_corr_series.values)))
        best_lag = lag_corr_series.index[best_idx]
        best_lag_corr = lag_corr_series.iloc[best_idx]
        best_lag_td = slot * best_lag

        print(f"\n[{phase_label}] Cross-correlation: best lag = {best_lag} slots "
              f"({best_lag_td}), r = {best_lag_corr:.4f}  (zero-lag r = {corr:.4f})")

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

    # Use twin axes so each series shows its own actual voltage values on
    # left/right sides, but both are visually aligned by spanning the same
    # number of standard deviations (z-score alignment without losing units).
    std1 = s1.std() if s1.std() > 0 else 1.0
    std2 = s2.std() if s2.std() > 0 else 1.0
    # k = half-range in units of std to show (covers the full data range + margin)
    k1 = (s1.max() - s1.min()) / (2 * std1) if std1 > 0 else 3.0
    k2 = (s2.max() - s2.min()) / (2 * std2) if std2 > 0 else 3.0
    k = max(k1, k2) * 1.05

    n_rows = 2 if (cross_corr and lag_corr_series is not None) else 1
    fig_h = 9 if n_rows == 2 else 5
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, fig_h),
                             gridspec_kw={"height_ratios": [3, 1] if n_rows == 2 else [1]})
    ax = axes[0] if n_rows == 2 else axes
    ax2 = ax.twinx()

    ax.plot(s1.index, s1.values, linewidth=0.8, label=label1, alpha=0.8, color="C0")
    ax2.plot(s2.index, s2.values, linewidth=0.8, label=label2, alpha=0.8, color="C1")

    ax.set_ylim(s1.mean() - k * std1, s1.mean() + k * std1)
    ax2.set_ylim(s2.mean() - k * std2, s2.mean() + k * std2)

    for start, end in neg_periods:
        ax.axvspan(start, end, color="red", alpha=0.1)

    ax.set_xlabel("Timestamp (CST)")
    ax.set_ylabel(f"{label1} Voltage", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2.set_ylabel(f"{label2} Voltage", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    title = f"{phase_label} — Pearson r = {corr:.4f}"
    if cross_corr and best_lag != 0:
        title += f"  |  best lag = {best_lag} slots  r = {best_lag_corr:.4f}"
    ax.set_title(title)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))

    if cross_corr and lag_corr_series is not None:
        ax_xc = axes[1]
        ax_xc.bar(lag_corr_series.index, lag_corr_series.values,
                  width=0.8, color="steelblue", alpha=0.7)
        ax_xc.axvline(best_lag, color="red", linewidth=1.2, linestyle="--",
                      label=f"best lag={best_lag} (r={best_lag_corr:.3f})")
        ax_xc.axvline(0, color="grey", linewidth=0.8, linestyle=":")
        ax_xc.axhline(0, color="black", linewidth=0.6)
        ax_xc.set_xlabel("Lag (slots)")
        ax_xc.set_ylabel("Pearson r")
        ax_xc.set_title("Cross-correlation by lag")
        ax_xc.legend(fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()

    if neg_periods:
        print(f"\n[{phase_label}] {len(neg_periods)} negatively correlated period(s):")
        for i, (start, end) in enumerate(neg_periods, 1):
            seg_corr = _corr(s1.loc[start:end], s2.loc[start:end])
            print(f"  {i}. {start}  →  {end}   (r = {seg_corr:.3f})")
    else:
        print(f"\n[{phase_label}] No negatively correlated periods found.")

    plt.show()
    if cross_corr and lag_corr_series is not None:
        return {"r": corr, "best_lag": best_lag, "best_lag_r": best_lag_corr,
                "lag_corr": lag_corr_series}
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
                                   plot=True):
    """
    Three-week load forecast per feeder.

    Week 1  — fit both a periodic (hour, is_weekend) profile and a
              temperature regression (linear + quadratic).
    Week 2  — validate both models; choose the winner by MAE.
    Weeks 1+2 — retrain the winning model on the combined window.
    Week 3  — held-out forecast / evaluation.

    Returns a dict keyed by feeder with:
        forecast_method            — "periodic" or "temperature"
        slope, intercept, train_r  — week-1 temperature model info
        val_mae_periodic           — validation MAE of periodic model
        val_mae_temperature        — validation MAE of best temperature model
        mae, rmse, forecast_r      — week-3 forecast quality (linear / periodic)
        mae_quad, rmse_quad, ...   — week-3 quadratic metrics
        coef_quad                  — quadratic coefficients (or [])
        week3                      — DataFrame of week-3 rows with forecast columns
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
        t_start  = fdf[time_col].min()
        t_split1 = t_start + pd.Timedelta(days=7)
        t_split2 = t_start + pd.Timedelta(days=14)
        if fdf[time_col].max() < t_split2:
            print(f"  [{feeder}] less than 3 weeks of data — skipped")
            continue
        train  = fdf[fdf[time_col] < t_split1].copy()
        val    = fdf[(fdf[time_col] >= t_split1) & (fdf[time_col] < t_split2)].copy()
        test   = fdf[fdf[time_col] >= t_split2].copy()
        w1 = f"{t_start.date()} – {(t_split1 - pd.Timedelta(days=1)).date()}"
        w2 = f"{t_split1.date()} – {(t_split2 - pd.Timedelta(days=1)).date()}"
        w3 = f"{t_split2.date()} – {fdf[time_col].max().date()}"
        fdf["_week"] = fdf[time_col].dt.to_period("W")  # for scatter coloring

        if len(train) < 2 or len(val) < 2 or test.empty:
            print(f"  [{feeder}] insufficient data in one of the windows — skipped")
            continue

        def _m(ya, yf):
            r = ya - yf
            with np.errstate(divide="ignore", invalid="ignore"):
                return (float(np.mean(np.abs(r))), float(np.sqrt(np.mean(r ** 2))),
                        float(np.corrcoef(ya, yf)[0, 1]))

        # ── step 1: fit both models on week 1 ───────────────────────────────
        x_train = train[temp_col].values
        y_train = train[load_col].values
        with np.errstate(divide="ignore", invalid="ignore"):
            train_r = float(np.corrcoef(x_train, y_train)[0, 1])
        if np.isnan(train_r):
            train_r = 0.0

        # periodic profile from week 1
        _tr = train.copy()
        _tr["_h"]   = _tr[time_col].dt.hour
        _tr["_dow"] = _tr[time_col].dt.dayofweek
        profile_w1   = _tr.groupby(["_h", "_dow"])[load_col].mean()
        hour_mean_w1 = _tr.groupby("_h")[load_col].mean()

        # temperature model from week 1
        coef1_w1 = np.polyfit(x_train, y_train, 1)
        coef2_w1 = np.polyfit(x_train, y_train, 2)

        # (hour, dow) residual profiles for hybrid temperature validation
        _tr2 = _tr.copy()
        _tr2["_rl"] = _tr2[load_col] - np.polyval(coef1_w1, _tr2[temp_col].values)
        _tr2["_rq"] = _tr2[load_col] - np.polyval(coef2_w1, _tr2[temp_col].values)
        rp1_w1 = _tr2.groupby(["_h", "_dow"])["_rl"].mean()
        rp2_w1 = _tr2.groupby(["_h", "_dow"])["_rq"].mean()
        rh1_w1 = _tr2.groupby("_h")["_rl"].mean()
        rh2_w1 = _tr2.groupby("_h")["_rq"].mean()

        def _resid(r, rp, rh):
            return rp.get((r["_h"], r["_dow"]), rh.get(r["_h"], 0.0))

        # ── step 2: validate on week 2 ──────────────────────────────────────
        _val = val.copy()
        _val["_h"]   = _val[time_col].dt.hour
        _val["_dow"] = _val[time_col].dt.dayofweek
        _val["_fc"] = _val.apply(
            lambda r: profile_w1.get((r["_h"], r["_dow"]),
                                     hour_mean_w1.get(r["_h"], float("nan"))), axis=1)
        y_val_act  = val[load_col].values
        y_val_per  = _val["_fc"].values
        y_val_lin  = (np.polyval(coef1_w1, val[temp_col].values)
                      + _val.apply(lambda r: _resid(r, rp1_w1, rh1_w1), axis=1).values)
        y_val_quad = (np.polyval(coef2_w1, val[temp_col].values)
                      + _val.apply(lambda r: _resid(r, rp2_w1, rh2_w1), axis=1).values)

        val_mae_periodic    = float(np.mean(np.abs(y_val_act - y_val_per)))
        val_mae_lin         = float(np.mean(np.abs(y_val_act - y_val_lin)))
        val_mae_quad        = float(np.mean(np.abs(y_val_act - y_val_quad)))
        val_mae_temperature = min(val_mae_lin, val_mae_quad)

        if len(val) < 4:
            use_periodic = abs(train_r) < 0.3
            print(f"  [{feeder}] validation window too small — falling back to r-threshold heuristic")
        else:
            use_periodic = val_mae_periodic <= val_mae_temperature

        # ── step 3: retrain winner on weeks 1+2 ─────────────────────────────
        train12 = fdf[fdf[time_col] < t_split2].copy()

        if use_periodic:
            _tr12 = train12.copy()
            _tr12["_h"]   = _tr12[time_col].dt.hour
            _tr12["_dow"] = _tr12[time_col].dt.dayofweek
            profile_final   = _tr12.groupby(["_h", "_dow"])[load_col].mean()
            hour_mean_final = _tr12.groupby("_h")[load_col].mean()
        else:
            x12 = train12[temp_col].values
            y12 = train12[load_col].values
            coef1_final = np.polyfit(x12, y12, 1)
            coef2_final = np.polyfit(x12, y12, 2)
            slope_final, intercept_final = float(coef1_final[0]), float(coef1_final[1])
            _tr12r = train12.copy()
            _tr12r["_h"]   = _tr12r[time_col].dt.hour
            _tr12r["_dow"] = _tr12r[time_col].dt.dayofweek
            _tr12r["_rl"]  = _tr12r[load_col] - np.polyval(coef1_final, _tr12r[temp_col].values)
            _tr12r["_rq"]  = _tr12r[load_col] - np.polyval(coef2_final, _tr12r[temp_col].values)
            rp1_f = _tr12r.groupby(["_h", "_dow"])["_rl"].mean()
            rp2_f = _tr12r.groupby(["_h", "_dow"])["_rq"].mean()
            rh1_f = _tr12r.groupby("_h")["_rl"].mean()
            rh2_f = _tr12r.groupby("_h")["_rq"].mean()

        # ── step 4: forecast week 3 ──────────────────────────────────────────
        if use_periodic:
            _te = test.copy()
            _te["_h"]   = _te[time_col].dt.hour
            _te["_dow"] = _te[time_col].dt.dayofweek
            _te["_fc"] = _te.apply(
                lambda r: profile_final.get((r["_h"], r["_dow"]),
                                            hour_mean_final.get(r["_h"], float("nan"))), axis=1)
            _te["load_forecast"]      = _te["_fc"]
            _te["load_forecast_quad"] = _te["_fc"]
            _te.drop(columns=["_h", "_dow", "_fc"], inplace=True)

            y_actual = _te[load_col].values
            mae, rmse, forecast_r = _m(y_actual, _te["load_forecast"].values)

            print(f"  [{feeder}]  val: per_MAE={val_mae_periodic:.4g}  temp_MAE={val_mae_temperature:.4g}  → periodic")
            print(f"           week 3 ({w3})  MAE={mae:.4g}  RMSE={rmse:.4g}  r={forecast_r:.4f}")

            results[feeder] = {
                "forecast_method": "periodic",
                "slope": float("nan"), "intercept": float("nan"), "train_r": train_r,
                "val_mae_periodic": val_mae_periodic, "val_mae_temperature": val_mae_temperature,
                "mae": mae, "rmse": rmse, "forecast_r": forecast_r,
                "mae_quad": mae, "rmse_quad": rmse, "forecast_r_quad": forecast_r,
                "coef_quad": [], "week3": _te,
            }

            if not plot:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(
                f"Feeder: {feeder}  [periodic  val_per={val_mae_periodic:.4g}  val_temp={val_mae_temperature:.4g}]",
                fontsize=12)

            hours = np.arange(24)

            def _plot_profile(ax_, df_, title_):
                _dow_cmap  = plt.get_cmap("tab10")
                _dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                for dow in range(7):
                    sub = df_[df_[time_col].dt.dayofweek == dow]
                    if sub.empty:
                        continue
                    sub = sub.copy()
                    sub["_h"] = sub[time_col].dt.hour
                    hm = sub.groupby("_h")[load_col].mean()
                    ax_.plot(hours, [hm.get(h, np.nan) for h in hours],
                             "o-", color=_dow_cmap(dow), linewidth=1.4, markersize=4,
                             label=_dow_names[dow])
                ax_.set_xlabel("hour of day")
                ax_.set_ylabel(load_col)
                ax_.set_title(title_)
                ax_.legend(fontsize=7, ncol=2)
                ax_.set_xticks(range(0, 24, 3))

            _plot_profile(axes[0, 0], train12, f"Weeks 1+2 load profile (final)  ({w1} / {w2})")

            ax2 = axes[0, 1]
            ts_12 = train12.set_index(time_col)[[load_col, temp_col]].sort_index().resample("1h").mean()
            ax2.plot(ts_12.index, ts_12[load_col], color="steelblue",
                     linewidth=0.7, label=load_col, alpha=0.85)
            ax2.set_ylabel(load_col, color="steelblue")
            ax2.tick_params(axis="y", labelcolor="steelblue")
            ax_r2 = ax2.twinx()
            ax_r2.plot(ts_12.index, ts_12[temp_col], color="darkorange",
                       linewidth=0.7, label=temp_col, alpha=0.85)
            ax_r2.set_ylabel(temp_col, color="darkorange")
            ax_r2.tick_params(axis="y", labelcolor="darkorange")
            lines2 = ax2.get_lines() + ax_r2.get_lines()
            ax2.legend(lines2, [l.get_label() for l in lines2], fontsize=8, loc="upper left")
            ax2.set_title(f"Weeks 1+2 — Load & Temperature  ({w1} / {w2})")

            _plot_profile(axes[1, 0], train, f"Week-1 profile  ({w1})")

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
            ax4.set_title(f"Week 3 forecast vs actual  ({w3})")

            fig.autofmt_xdate()
            fig.tight_layout()
            plt.show()
            continue

        # ── temperature-based branch ─────────────────────────────────────────
        test = test.copy()
        test["_h"]   = test[time_col].dt.hour
        test["_dow"] = test[time_col].dt.dayofweek
        x_test = test[temp_col].values
        y_actual = test[load_col].values
        y_fc_lin  = (np.polyval(coef1_final, x_test)
                     + test.apply(lambda r: _resid(r, rp1_f, rh1_f), axis=1).values)
        y_fc_quad = (np.polyval(coef2_final, x_test)
                     + test.apply(lambda r: _resid(r, rp2_f, rh2_f), axis=1).values)
        test["load_forecast"]      = y_fc_lin
        test["load_forecast_quad"] = y_fc_quad
        test.drop(columns=["_h", "_dow"], inplace=True)

        slope, intercept = slope_final, intercept_final
        mae,  rmse,  forecast_r  = _m(y_actual, y_fc_lin)
        mae2, rmse2, forecast_r2 = _m(y_actual, y_fc_quad)

        print(f"  [{feeder}]  val: per_MAE={val_mae_periodic:.4g}  temp_MAE={val_mae_temperature:.4g}  → temperature")
        print(f"           week 3 ({w3})  linear:    MAE={mae:.4g}  RMSE={rmse:.4g}  r={forecast_r:.4f}")
        print(f"                          quadratic: MAE={mae2:.4g}  RMSE={rmse2:.4g}  r={forecast_r2:.4f}")

        results[feeder] = {
            "forecast_method": "temperature",
            "slope": slope, "intercept": intercept, "train_r": train_r,
            "val_mae_periodic": val_mae_periodic, "val_mae_temperature": val_mae_temperature,
            "mae": mae, "rmse": rmse, "forecast_r": forecast_r,
            "mae_quad": mae2, "rmse_quad": rmse2, "forecast_r_quad": forecast_r2,
            "coef_quad": coef2_final.tolist(), "week3": test,
        }

        if not plot:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Feeder: {feeder}  [val_per={val_mae_periodic:.4g}  val_temp={val_mae_temperature:.4g}]",
            fontsize=12)

        all_x = fdf[temp_col].values
        all_y = fdf[load_col].values
        overall_coef = np.polyfit(all_x, all_y, 1)
        overall_r    = float(np.corrcoef(all_x, all_y)[0, 1])
        xline_all    = np.linspace(all_x.min(), all_x.max(), 200)

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

        ax2 = axes[0, 1]
        ts_12 = train12.set_index(time_col)[[load_col, temp_col]].sort_index().resample("1h").mean()
        color_load, color_temp = "steelblue", "darkorange"
        ax2.plot(ts_12.index, ts_12[load_col], color=color_load,
                 linewidth=0.7, label=load_col, alpha=0.85)
        ax2.set_ylabel(load_col, color=color_load)
        ax2.tick_params(axis="y", labelcolor=color_load)
        ax_r2 = ax2.twinx()
        ax_r2.plot(ts_12.index, ts_12[temp_col], color=color_temp,
                   linewidth=0.7, label=temp_col, alpha=0.85)
        ax_r2.set_ylabel(temp_col, color=color_temp)
        ax_r2.tick_params(axis="y", labelcolor=color_temp)
        lines2 = ax2.get_lines() + ax_r2.get_lines()
        ax2.legend(lines2, [l.get_label() for l in lines2], fontsize=8, loc="upper left")
        ax2.set_title(f"Weeks 1+2 — Load & Temperature  ({w1} / {w2})")

        ax3 = axes[1, 0]
        ax3.scatter(x_train, y_train, s=8, alpha=0.45, linewidths=0,
                    label=f"week 1 data  (r={train_r:.3f})")
        xline_w1 = np.linspace(x_train.min(), x_train.max(), 200)
        ax3.plot(xline_w1, np.polyval(coef1_w1, xline_w1), color="crimson",
                 linewidth=1.5, label=f"linear  slope={coef1_w1[0]:.3g}")
        ax3.plot(xline_w1, np.polyval(coef2_w1, xline_w1), color="darkorange",
                 linewidth=1.5, linestyle="--", label="quadratic")
        ax3.set_xlabel(temp_col)
        ax3.set_ylabel(load_col)
        ax3.set_title(f"Week 1 initial fit  ({w1})")
        ax3.legend(fontsize=8)

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
        ax4.set_title(f"Week 3 forecast vs actual  ({w3})")

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
                              plot=True):
    """
    Offset-aware three-week load forecast per feeder.

    Week 1  — fit periodic profile + temperature model (with peak-offset shift).
    Week 2  — validate both models; choose winner by MAE.
    Weeks 1+2 — retrain winner on combined window.
    Week 3  — held-out forecast / evaluation.

    Returns a dict keyed by feeder containing:
        forecast_method              — "periodic" or "temperature"
        offset_h                     — median peak offset used (0 for periodic)
        slope, intercept, train_r    — week-1 temperature model info
        val_mae_periodic             — validation MAE of periodic model
        val_mae_temperature          — validation MAE of best temperature model
        mae, rmse, forecast_r        — week-3 metrics (linear / periodic)
        mae_quad, rmse_quad, ...     — week-3 quadratic metrics
        coef_quad                    — quadratic coefficients (or [])
        week3                        — DataFrame with actual load and forecast columns
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
        t_start  = fdf[time_col].min()
        t_split1 = t_start + pd.Timedelta(days=7)
        t_split2 = t_start + pd.Timedelta(days=14)
        if fdf[time_col].max() < t_split2:
            print(f"  [{feeder}] less than 3 weeks of data — skipped")
            continue

        train_orig = fdf[fdf[time_col] < t_split1].copy()
        val_orig   = fdf[(fdf[time_col] >= t_split1) & (fdf[time_col] < t_split2)].copy()
        test_orig  = fdf[fdf[time_col] >= t_split2].copy()
        w1 = f"{t_start.date()} – {(t_split1 - pd.Timedelta(days=1)).date()}"
        w2 = f"{t_split1.date()} – {(t_split2 - pd.Timedelta(days=1)).date()}"
        w3 = f"{t_split2.date()} – {fdf[time_col].max().date()}"

        if len(train_orig) < 2 or len(val_orig) < 2 or test_orig.empty:
            print(f"  [{feeder}] insufficient data in one of the windows — skipped")
            continue

        def _m(ya, yf):
            r = ya - yf
            with np.errstate(divide="ignore", invalid="ignore"):
                return (float(np.mean(np.abs(r))), float(np.sqrt(np.mean(r ** 2))),
                        float(np.corrcoef(ya, yf)[0, 1]))

        import matplotlib.gridspec as gridspec
        import matplotlib.dates as mdates

        # ── step 1: fit both models on week 1 ───────────────────────────────

        # periodic profile from raw week-1 load
        fload_w1p = load_df[
            (load_df[feeder_col] == feeder) &
            (load_df[time_col] >= t_start) &
            (load_df[time_col] < t_split1)
        ].copy()
        fload_w1p[load_col] = fload_w1p[load_col].astype(float)
        fload_w1p["_h"]   = fload_w1p[time_col].dt.hour
        fload_w1p["_dow"] = fload_w1p[time_col].dt.dayofweek
        profile_w1   = fload_w1p.groupby(["_h", "_dow"])[load_col].mean()
        hour_mean_w1 = fload_w1p.groupby("_h")[load_col].mean()

        # raw correlation for fallback heuristic
        with np.errstate(divide="ignore", invalid="ignore"):
            train_r_check = float(np.corrcoef(train_orig[temp_col].values,
                                              train_orig[load_col].values)[0, 1]) \
                            if len(train_orig) >= 2 else 0.0
        if np.isnan(train_r_check):
            train_r_check = 0.0

        # compute offset from week-1 merged data
        _tr1 = train_orig.copy()
        _tr1["_date"] = _tr1[time_col].dt.date
        _tr1["_hour"] = _tr1[time_col].dt.hour
        day_offsets = []
        for _, day in _tr1.groupby("_date"):
            if day.empty:
                continue
            pl = int(day.loc[day[load_col].idxmax(), "_hour"])
            pt = int(day.loc[day[temp_col].idxmax(), "_hour"])
            day_offsets.append(pl - pt)
        if not day_offsets:
            print(f"  [{feeder}] could not compute peak offset — skipped")
            continue
        offset_h_w1 = int(round(float(np.median(day_offsets))))

        # shift week-1 load, re-merge with temp, fit temperature model
        fload_w1 = load_df[
            (load_df[feeder_col] == feeder) &
            (load_df[time_col] >= t_start) &
            (load_df[time_col] < t_split1)
        ].copy()
        fload_w1_shifted = fload_w1.copy()
        fload_w1_shifted[time_col] = fload_w1_shifted[time_col] - pd.Timedelta(hours=offset_h_w1)
        train_shifted = pd.merge(fload_w1_shifted, temp_df, on=[time_col, poi_col],
                                 how="inner", suffixes=("_load", "_temp"))
        train_shifted[load_col] = train_shifted[load_col].astype(float)
        train_shifted[temp_col] = train_shifted[temp_col].astype(float)
        train_shifted = train_shifted[train_shifted[load_col].notna() & train_shifted[temp_col].notna()]
        if len(train_shifted) < 2:
            print(f"  [{feeder}] insufficient overlap after shifting week-1 — skipped")
            continue

        train_agg1 = (train_shifted
                      .set_index(time_col)[[load_col, temp_col]]
                      .resample("1h").mean().dropna().reset_index())
        x_train1 = train_agg1[temp_col].values
        y_train1 = train_agg1[load_col].values
        coef1_w1 = np.polyfit(x_train1, y_train1, 1)
        coef2_w1 = np.polyfit(x_train1, y_train1, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            train_r = float(np.corrcoef(x_train1, y_train1)[0, 1])
        if np.isnan(train_r):
            train_r = 0.0

        # (hour, dow) residual profiles using original-time coords
        orig_times_w1 = train_agg1[time_col] + pd.Timedelta(hours=offset_h_w1)
        _tagg1r = train_agg1.copy()
        _tagg1r["_h"]   = orig_times_w1.dt.hour
        _tagg1r["_dow"] = orig_times_w1.dt.dayofweek
        _tagg1r["_rl"]  = _tagg1r[load_col] - np.polyval(coef1_w1, _tagg1r[temp_col].values)
        _tagg1r["_rq"]  = _tagg1r[load_col] - np.polyval(coef2_w1, _tagg1r[temp_col].values)
        rp1_w1 = _tagg1r.groupby(["_h", "_dow"])["_rl"].mean()
        rp2_w1 = _tagg1r.groupby(["_h", "_dow"])["_rq"].mean()
        rh1_w1 = _tagg1r.groupby("_h")["_rl"].mean()
        rh2_w1 = _tagg1r.groupby("_h")["_rq"].mean()

        def _resid(r, rp, rh):
            return rp.get((r["_h"], r["_dow"]), rh.get(r["_h"], 0.0))

        # ── step 2: validate on week 2 ──────────────────────────────────────

        # periodic validation
        fload_w2v = load_df[
            (load_df[feeder_col] == feeder) &
            (load_df[time_col] >= t_split1) &
            (load_df[time_col] < t_split2)
        ].copy()
        fload_w2v[load_col] = fload_w2v[load_col].astype(float)
        fload_w2v["_h"]   = fload_w2v[time_col].dt.hour
        fload_w2v["_dow"] = fload_w2v[time_col].dt.dayofweek
        fload_w2v["_fc"] = fload_w2v.apply(
            lambda r: profile_w1.get((r["_h"], r["_dow"]),
                                     hour_mean_w1.get(r["_h"], float("nan"))), axis=1)
        fload_w2v["_fc_per"] = fload_w2v["_fc"]
        fload_w2v.drop(columns=["_h", "_dow", "_fc"], inplace=True)
        val_per = (fload_w2v.set_index(time_col)[[load_col, "_fc_per"]]
                   .resample("1h").mean().dropna().reset_index())

        # temperature validation using week-1 coefs + offset
        ftemp_w2v = temp_df[
            (temp_df[time_col] >= t_split1) & (temp_df[time_col] < t_split2)
        ].copy()
        ftemp_w2v[temp_col] = ftemp_w2v[temp_col].astype(float)
        ftemp_w2v["_h"]   = (ftemp_w2v[time_col] + pd.Timedelta(hours=offset_h_w1)).dt.hour
        ftemp_w2v["_dow"] = (ftemp_w2v[time_col] + pd.Timedelta(hours=offset_h_w1)).dt.dayofweek
        ftemp_w2v["_fc_lin"]  = (np.polyval(coef1_w1, ftemp_w2v[temp_col].values)
                                  + ftemp_w2v.apply(lambda r: _resid(r, rp1_w1, rh1_w1), axis=1).values)
        ftemp_w2v["_fc_quad"] = (np.polyval(coef2_w1, ftemp_w2v[temp_col].values)
                                  + ftemp_w2v.apply(lambda r: _resid(r, rp2_w1, rh2_w1), axis=1).values)
        ftemp_w2v[time_col] = ftemp_w2v[time_col] + pd.Timedelta(hours=offset_h_w1)
        ftemp_w2v.drop(columns=["_h", "_dow"], inplace=True)

        fload_w2a = load_df[
            (load_df[feeder_col] == feeder) &
            (load_df[time_col] >= t_split1) &
            (load_df[time_col] < t_split2)
        ].copy()
        fload_w2a[load_col] = fload_w2a[load_col].astype(float)
        val_temp_raw = pd.merge(
            fload_w2a[[time_col, feeder_col, load_col]],
            ftemp_w2v[[time_col, "_fc_lin", "_fc_quad"]],
            on=time_col, how="inner"
        ).dropna(subset=[load_col, "_fc_lin"])
        val_temp = (val_temp_raw
                    .set_index(time_col)[[load_col, "_fc_lin", "_fc_quad"]]
                    .resample("1h").mean().dropna().reset_index())

        if val_per.empty or val_temp.empty:
            use_periodic = abs(train_r_check) < 0.3
            val_mae_periodic    = float("nan")
            val_mae_temperature = float("nan")
            print(f"  [{feeder}] validation data missing — falling back to r-threshold heuristic")
        else:
            val_mae_periodic = float(np.mean(np.abs(
                val_per[load_col].values - val_per["_fc_per"].values)))
            val_mae_lin  = float(np.mean(np.abs(
                val_temp[load_col].values - val_temp["_fc_lin"].values)))
            val_mae_quad = float(np.mean(np.abs(
                val_temp[load_col].values - val_temp["_fc_quad"].values)))
            val_mae_temperature = min(val_mae_lin, val_mae_quad)
            if len(val_per) < 4:
                use_periodic = abs(train_r_check) < 0.3
                print(f"  [{feeder}] validation window too small — falling back to r-threshold heuristic")
            else:
                use_periodic = val_mae_periodic <= val_mae_temperature

        # ── step 3: retrain winner on weeks 1+2 ─────────────────────────────
        if use_periodic:
            fload_w12p = load_df[
                (load_df[feeder_col] == feeder) &
                (load_df[time_col] >= t_start) &
                (load_df[time_col] < t_split2)
            ].copy()
            fload_w12p[load_col] = fload_w12p[load_col].astype(float)
            fload_w12p["_h"]   = fload_w12p[time_col].dt.hour
            fload_w12p["_dow"] = fload_w12p[time_col].dt.dayofweek
            profile_final   = fload_w12p.groupby(["_h", "_dow"])[load_col].mean()
            hour_mean_final = fload_w12p.groupby("_h")[load_col].mean()
            offset_h = 0
        else:
            # recompute offset from weeks 1+2 merged data
            train12_orig = fdf[fdf[time_col] < t_split2].copy()
            train12_orig["_date"] = train12_orig[time_col].dt.date
            train12_orig["_hour"] = train12_orig[time_col].dt.hour
            day_offsets12 = []
            for _, day in train12_orig.groupby("_date"):
                if day.empty:
                    continue
                pl = int(day.loc[day[load_col].idxmax(), "_hour"])
                pt = int(day.loc[day[temp_col].idxmax(), "_hour"])
                day_offsets12.append(pl - pt)
            offset_h = int(round(float(np.median(day_offsets12)))) if day_offsets12 else offset_h_w1

            fload_w12 = load_df[
                (load_df[feeder_col] == feeder) &
                (load_df[time_col] >= t_start) &
                (load_df[time_col] < t_split2)
            ].copy()
            fload_w12_shifted = fload_w12.copy()
            fload_w12_shifted[time_col] = fload_w12_shifted[time_col] - pd.Timedelta(hours=offset_h)
            train12_shifted = pd.merge(fload_w12_shifted, temp_df, on=[time_col, poi_col],
                                       how="inner", suffixes=("_load", "_temp"))
            train12_shifted[load_col] = train12_shifted[load_col].astype(float)
            train12_shifted[temp_col] = train12_shifted[temp_col].astype(float)
            train12_shifted = train12_shifted[
                train12_shifted[load_col].notna() & train12_shifted[temp_col].notna()]
            if len(train12_shifted) < 2:
                print(f"  [{feeder}] insufficient overlap after shifting weeks 1+2 — skipped")
                continue

            train_agg12 = (train12_shifted
                           .set_index(time_col)[[load_col, temp_col]]
                           .resample("1h").mean().dropna().reset_index())
            x12 = train_agg12[temp_col].values
            y12 = train_agg12[load_col].values
            coef1_final = np.polyfit(x12, y12, 1)
            coef2_final = np.polyfit(x12, y12, 2)
            slope_final, intercept_final = float(coef1_final[0]), float(coef1_final[1])
            with np.errstate(divide="ignore", invalid="ignore"):
                train12_r = float(np.corrcoef(x12, y12)[0, 1])
            orig_times_12 = train_agg12[time_col] + pd.Timedelta(hours=offset_h)
            _tagg12r = train_agg12.copy()
            _tagg12r["_h"]   = orig_times_12.dt.hour
            _tagg12r["_dow"] = orig_times_12.dt.dayofweek
            _tagg12r["_rl"]  = _tagg12r[load_col] - np.polyval(coef1_final, _tagg12r[temp_col].values)
            _tagg12r["_rq"]  = _tagg12r[load_col] - np.polyval(coef2_final, _tagg12r[temp_col].values)
            rp1_f = _tagg12r.groupby(["_h", "_dow"])["_rl"].mean()
            rp2_f = _tagg12r.groupby(["_h", "_dow"])["_rq"].mean()
            rh1_f = _tagg12r.groupby("_h")["_rl"].mean()
            rh2_f = _tagg12r.groupby("_h")["_rq"].mean()

        # ── step 4: forecast week 3 ──────────────────────────────────────────
        if use_periodic:
            fload_w3p = load_df[
                (load_df[feeder_col] == feeder) &
                (load_df[time_col] >= t_split2)
            ].copy()
            fload_w3p[load_col] = fload_w3p[load_col].astype(float)
            fload_w3p["_h"]   = fload_w3p[time_col].dt.hour
            fload_w3p["_dow"] = fload_w3p[time_col].dt.dayofweek
            fload_w3p["_fc"] = fload_w3p.apply(
                lambda r: profile_final.get((r["_h"], r["_dow"]),
                                            hour_mean_final.get(r["_h"], float("nan"))), axis=1)
            fload_w3p["load_forecast"]      = fload_w3p["_fc"]
            fload_w3p["load_forecast_quad"] = fload_w3p["_fc"]
            fload_w3p.drop(columns=["_h", "_dow", "_fc"], inplace=True)
            week3 = (fload_w3p.set_index(time_col)[[load_col, "load_forecast", "load_forecast_quad"]]
                     .resample("1h").mean().dropna().reset_index())

            if week3.empty:
                print(f"  [{feeder}] no week-3 load data — skipped")
                continue

            y_actual = week3[load_col].values
            mae, rmse, forecast_r = _m(y_actual, week3["load_forecast"].values)

            print(f"  [{feeder}]  val: per_MAE={val_mae_periodic:.4g}  temp_MAE={val_mae_temperature:.4g}  → periodic")
            print(f"           week 3 ({w3})  MAE={mae:.4g}  RMSE={rmse:.4g}  r={forecast_r:.4f}")

            results[feeder] = {
                "forecast_method": "periodic",
                "offset_h": 0, "slope": float("nan"), "intercept": float("nan"),
                "train_r": train_r, "val_mae_periodic": val_mae_periodic,
                "val_mae_temperature": val_mae_temperature,
                "mae": mae, "rmse": rmse, "forecast_r": forecast_r,
                "mae_quad": mae, "rmse_quad": rmse, "forecast_r_quad": forecast_r,
                "coef_quad": [], "week3": week3,
            }

            if not plot:
                continue

            fig = plt.figure(figsize=(14, 10))
            fig.suptitle(
                f"Feeder: {feeder}  [periodic  val_per={val_mae_periodic:.4g}  val_temp={val_mae_temperature:.4g}]",
                fontsize=12)
            gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
            ax_prof = fig.add_subplot(gs[0, 0])
            ax_ts12 = fig.add_subplot(gs[0, 1])
            ax_fc   = fig.add_subplot(gs[1, :])

            # top-left: weeks 1+2 hourly profile
            fload_w12p_plot = load_df[
                (load_df[feeder_col] == feeder) &
                (load_df[time_col] >= t_start) &
                (load_df[time_col] < t_split2)
            ].copy()
            hours = np.arange(24)
            _dow_cmap  = plt.get_cmap("tab10")
            _dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            for dow in range(7):
                sub = fload_w12p_plot[fload_w12p_plot[time_col].dt.dayofweek == dow]
                if sub.empty:
                    continue
                sub = sub.copy()
                sub["_h"] = sub[time_col].dt.hour
                hm = sub.groupby("_h")[load_col].mean()
                ax_prof.plot(hours, [hm.get(h, np.nan) for h in hours],
                             "o-", color=_dow_cmap(dow), linewidth=1.4, markersize=4,
                             label=_dow_names[dow])
            ax_prof.set_xlabel("hour of day")
            ax_prof.set_ylabel(load_col)
            ax_prof.set_title(f"Weeks 1+2 hourly load profile  ({w1} / {w2})")
            ax_prof.legend(fontsize=7, ncol=2)
            ax_prof.set_xticks(range(0, 24, 3))

            # top-right: weeks 1+2 load time series
            _load_w12 = load_df[
                (load_df[feeder_col] == feeder) &
                (load_df[time_col] >= t_start) &
                (load_df[time_col] < t_split2)
            ].copy()
            ts_load_12 = (_load_w12.set_index(time_col)[load_col]
                          .groupby(level=0).mean().resample("1h").mean())
            ax_ts12.plot(ts_load_12.index, ts_load_12.values, color="steelblue",
                         linewidth=0.7, label=load_col, alpha=0.85)
            ax_ts12.set_ylabel(load_col)
            ax_ts12.set_title(f"Weeks 1+2 — Load  ({w1} / {w2})")
            ax_ts12.legend(fontsize=8)
            ax_ts12.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            plt.setp(ax_ts12.get_xticklabels(), rotation=30, ha="right")

            # bottom: week-3 actual vs periodic forecast
            w3s = (week3.set_index(time_col)[[load_col, "load_forecast"]]
                   .resample("1h").mean())
            _ts_temp_fc = (temp_df[temp_df[time_col] >= t_split2]
                           .set_index(time_col)[temp_col].astype(float)
                           .groupby(level=0).mean().resample("1h").mean())
            ax_fc.plot(w3s.index, w3s[load_col], color="steelblue", linewidth=0.9,
                       label="actual", alpha=0.9)
            ax_fc.plot(w3s.index, w3s["load_forecast"], color="crimson", linewidth=0.9,
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
            ax_fc.set_title(f"Week 3 — forecast vs actual  ({w3})")
            ax_fc.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            plt.setp(ax_fc.get_xticklabels(), rotation=30, ha="right")

            fig.tight_layout()
            plt.show()
            continue

        # ── temperature branch: forecast week 3 ─────────────────────────────
        ftemp_w3 = temp_df[temp_df[time_col] >= t_split2].copy()
        ftemp_w3[temp_col] = ftemp_w3[temp_col].astype(float)
        ftemp_w3["_h"]   = (ftemp_w3[time_col] + pd.Timedelta(hours=offset_h)).dt.hour
        ftemp_w3["_dow"] = (ftemp_w3[time_col] + pd.Timedelta(hours=offset_h)).dt.dayofweek
        ftemp_w3["load_forecast"]      = (np.polyval(coef1_final, ftemp_w3[temp_col].values)
                                          + ftemp_w3.apply(lambda r: _resid(r, rp1_f, rh1_f), axis=1).values)
        ftemp_w3["load_forecast_quad"] = (np.polyval(coef2_final, ftemp_w3[temp_col].values)
                                          + ftemp_w3.apply(lambda r: _resid(r, rp2_f, rh2_f), axis=1).values)
        ftemp_w3[time_col] = ftemp_w3[time_col] + pd.Timedelta(hours=offset_h)
        ftemp_w3.drop(columns=["_h", "_dow"], inplace=True)

        fload_w3 = load_df[
            (load_df[feeder_col] == feeder) &
            (load_df[time_col] >= t_split2)
        ].copy()
        fload_w3[load_col] = fload_w3[load_col].astype(float)

        week3_raw = pd.merge(
            fload_w3[[time_col, feeder_col, load_col]],
            ftemp_w3[[time_col, "load_forecast", "load_forecast_quad"]],
            on=time_col, how="inner"
        ).dropna(subset=[load_col, "load_forecast", "load_forecast_quad"])

        if week3_raw.empty:
            print(f"  [{feeder}] no overlapping records for week-3 comparison — skipped")
            continue

        week3 = (week3_raw
                 .set_index(time_col)[[load_col, "load_forecast", "load_forecast_quad"]]
                 .resample("1h").mean().dropna().reset_index())

        y_actual  = week3[load_col].values
        y_fc_lin  = week3["load_forecast"].values
        y_fc_quad = week3["load_forecast_quad"].values
        mae,  rmse,  forecast_r  = _m(y_actual, y_fc_lin)
        mae2, rmse2, forecast_r2 = _m(y_actual, y_fc_quad)
        slope, intercept = slope_final, intercept_final

        print(f"  [{feeder}]  val: per_MAE={val_mae_periodic:.4g}  temp_MAE={val_mae_temperature:.4g}  → temperature")
        print(f"           offset={offset_h:+d}h  train12_r={train12_r:.4f}  slope={slope:.4g}")
        print(f"           week-3  linear:    MAE={mae:.4g}  RMSE={rmse:.4g}  r={forecast_r:.4f}")
        print(f"                   quadratic: MAE={mae2:.4g}  RMSE={rmse2:.4g}  r={forecast_r2:.4f}")

        results[feeder] = {
            "forecast_method": "temperature",
            "offset_h": offset_h, "slope": slope, "intercept": intercept,
            "train_r": train_r, "val_mae_periodic": val_mae_periodic,
            "val_mae_temperature": val_mae_temperature,
            "mae": mae, "rmse": rmse, "forecast_r": forecast_r,
            "mae_quad": mae2, "rmse_quad": rmse2, "forecast_r_quad": forecast_r2,
            "coef_quad": coef2_final.tolist(), "week3": week3,
        }

        if not plot:
            continue

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            f"Feeder: {feeder}  (offset={offset_h:+d}h  val_per={val_mae_periodic:.4g}  val_temp={val_mae_temperature:.4g})",
            fontsize=12)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
        ax_scatter = fig.add_subplot(gs[0, 0])
        ax_ts12    = fig.add_subplot(gs[0, 1])
        ax_fc      = fig.add_subplot(gs[1, :])

        # top-left: shifted weeks 1+2 scatter + final fit
        ax_scatter.scatter(x12, y12, s=8, alpha=0.4, linewidths=0)
        xline = np.linspace(x12.min(), x12.max(), 200)
        ax_scatter.plot(xline, np.polyval(coef1_final, xline), color="crimson", linewidth=1.5,
                        label=f"linear  slope={slope:.3g}  r={train12_r:.3f}")
        ax_scatter.plot(xline, np.polyval(coef2_final, xline), color="darkorange", linewidth=1.5,
                        linestyle="--", label="quadratic")
        ax_scatter.set_xlabel(temp_col)
        ax_scatter.set_ylabel(load_col)
        ax_scatter.set_title(f"Weeks 1+2 (shifted {-offset_h:+d}h) — final fit  ({w1} / {w2})")
        ax_scatter.legend(fontsize=8)

        # top-right: weeks 1+2 dual-axis time series
        _load_w12 = load_df[
            (load_df[feeder_col] == feeder) &
            (load_df[time_col] >= t_start) &
            (load_df[time_col] < t_split2)
        ].copy()
        _temp_w12 = temp_df[
            (temp_df[time_col] >= t_start) & (temp_df[time_col] < t_split2)
        ].copy()
        _load_w12[load_col] = _load_w12[load_col].astype(float)
        _temp_w12[temp_col] = _temp_w12[temp_col].astype(float)
        ts_load = (_load_w12.set_index(time_col)[load_col]
                   .groupby(level=0).mean().resample("1h").mean())
        ts_temp = (_temp_w12.set_index(time_col)[temp_col]
                   .groupby(level=0).mean().resample("1h").mean())
        ax_ts12.plot(ts_load.index, ts_load.values, color="steelblue",
                     linewidth=0.7, label=load_col, alpha=0.85)
        ax_ts12.set_ylabel(load_col, color="steelblue")
        ax_ts12.tick_params(axis="y", labelcolor="steelblue")
        ax_ts12r = ax_ts12.twinx()
        ax_ts12r.plot(ts_temp.index, ts_temp.values, color="darkorange",
                      linewidth=0.7, label=temp_col, alpha=0.85)
        ax_ts12r.set_ylabel(temp_col, color="darkorange")
        ax_ts12r.tick_params(axis="y", labelcolor="darkorange")
        lines12 = ax_ts12.get_lines() + ax_ts12r.get_lines()
        ax_ts12.legend(lines12, [l.get_label() for l in lines12], fontsize=8, loc="upper left")
        ax_ts12.set_title(f"Weeks 1+2 — Load & Temperature  ({w1} / {w2})")
        ax_ts12.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.setp(ax_ts12.get_xticklabels(), rotation=30, ha="right")

        # bottom: week-3 actual vs forecast (full width)
        w3s = (week3.set_index(time_col)[[load_col, "load_forecast", "load_forecast_quad"]]
               .resample("1h").mean())
        _ts_temp_fc = (temp_df[temp_df[time_col] >= t_split2]
                       .set_index(time_col)[temp_col].astype(float)
                       .groupby(level=0).mean().resample("1h").mean())
        ax_fc.plot(w3s.index, w3s[load_col], color="steelblue", linewidth=0.9,
                   label="actual", alpha=0.9)
        ax_fc.plot(w3s.index, w3s["load_forecast"], color="crimson", linewidth=0.9,
                   linestyle="--",
                   label=f"linear  MAE={mae:.4g}  r={forecast_r:.3f}", alpha=0.9)
        ax_fc.plot(w3s.index, w3s["load_forecast_quad"], color="darkorange", linewidth=0.9,
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
        ax_fc.set_title(f"Week 3 — forecast vs actual  ({w3})")
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
