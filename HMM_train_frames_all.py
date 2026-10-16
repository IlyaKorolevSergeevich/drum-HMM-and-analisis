# hmm_train_frames_all.py
# -*- coding: utf-8 -*-
"""
Frame-level HMM training on combined blind + sighted data.

Groups are assigned by video name prefix:
    c3h*  -> blind
    cba*  -> sighted
The 'group' column in the input CSV is ignored and recomputed.

Pipeline:
  1. Load combined frame-features CSV
  2. Parse metadata (mouse, condition, stim_dir, group)
  3. Z-score per mouse + clip outliers
  4. Build per-video sequences (frame order)
  5. Fit GaussianHMM for K in K_RANGE
  6. Pick K by BIC (capped at K_MAX)
  7. Viterbi decode per frame
  8. Aggregate frames -> 1-s windows + per-video fractions
  9. Compare blind vs sighted (stats + PNG plots)
 10. Save emissions, transitions, timelines
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from scipy.stats import mannwhitneyu, chi2_contingency
from hmmlearn.hmm import GaussianHMM


# =========================================================
# CONFIG
# =========================================================

BASE_DIR   = r"C:\Users\90108\OneDrive\Рабочий стол\v\agouty"
FRAME_CSV  = os.path.join(BASE_DIR, "frame_features10", "frame_features10.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "hmm_results_frames10of10(5)")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HMM input features
HMM_FEATURES = [
    # Pose (normalized)
    "head_area_norm",
    "head_len_norm",
    "ear_asym_norm",
    # Orientation (circular)
    "head_angle_sin",
    "head_angle_cos",
    "head_body_diff_sin",
    "head_body_diff_cos",
    # Movement
    "hb_speed",
    "angular_speed",
]

# HMM search
K_RANGE   = [2, 3, 4, 5]
K_MAX     = 5
COV_TYPE  = "diag"
N_ITER    = 500
TOL       = 1e-4
SEED      = 42
N_SEEDS   = 10
MIN_COVAR = 0.05

CLIP_Z = 3.0

# Frame -> window aggregation
WINDOW_SECONDS = 1.0
FPS            = 30

# Group assignment by video name prefix
GROUP_BY_PREFIX = {
    "c3h": "blind",
    "cba": "sighted",
}


# =========================================================
# HELPERS
# =========================================================

def _group_from_video(video_name):
    if not isinstance(video_name, str):
        return "unknown"
    name = video_name.strip().lower()
    for prefix, group in GROUP_BY_PREFIX.items():
        if name.startswith(prefix.lower()):
            return group
    return "unknown"


def parse_meta(df):
    """Extract mouse / condition / stim_dir / group from video name.

    All four columns are always recomputed; any existing columns in
    the CSV are ignored.
    """
    df = df.copy()

    # Mouse: from the leading token of the video name
    parsed = df["video"].astype(str).str.split("_").str[0]
    parsed = parsed.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    df["mouse_parsed"] = parsed.fillna("unknown").astype(str)

    # Condition
    df["condition"] = np.select(
        [df["video"].str.contains("_black_", case=False, regex=False),
         df["video"].str.contains("_stim",  case=False, regex=False)],
        ["baseline", "stim"], default="other")

    # Stimulus direction
    df["stim_dir"] = np.select(
        [df["video"].str.endswith("_left"),
         df["video"].str.endswith("_right")],
        ["left", "right"], default="")

    # Group: strictly by video name prefix
    df["group"] = df["video"].map(_group_from_video)

    return df


def fit_best_model(X_all, lengths, K, n_seeds=N_SEEDS):
    best_model, best_ll = None, -np.inf
    for seed in range(n_seeds):
        m = GaussianHMM(
            n_components=K,
            covariance_type=COV_TYPE,
            n_iter=N_ITER,
            tol=TOL,
            min_covar=MIN_COVAR,
            random_state=seed,
        )
        try:
            m.fit(X_all, lengths)
            ll = m.score(X_all, lengths)
            if ll > best_ll:
                best_ll = ll
                best_model = m
        except Exception:
            continue
    return best_model, best_ll


def aggregate_window_states(df_states, K):
    df = df_states.copy()
    df["win_idx"] = (df["frame"] // int(WINDOW_SECONDS * FPS)).astype(int)
    out = (df
           .groupby(["mouse_parsed", "video", "group",
                     "condition", "stim_dir", "win_idx"])
           .apply(lambda g: pd.Series(
               {f"frac_state_{k}": float((g["hmm_state"] == k).mean())
                for k in range(K)}))
           .reset_index())
    out["win_start_s"] = out["win_idx"] * WINDOW_SECONDS
    return out


def aggregate_video_states(df_states, K):
    """Per-video aggregation."""
    rows = []
    for (mouse, video), g in df_states.groupby(
            ["mouse_parsed", "video"], sort=False):
        g = g.sort_values("frame")
        s = g["hmm_state"].values

        row = {
            "mouse_parsed": mouse,
            "video":        video,
            "group":        g["group"].iloc[0],
            "condition":    g["condition"].iloc[0],
            "stim_dir":     g["stim_dir"].iloc[0],
            "n_frames":     len(s),
        }
        for k in range(K):
            row[f"frac_state_{k}"] = float((s == k).mean())

        row["n_switches"] = int((np.diff(s) != 0).sum()) if len(s) > 1 else 0

        dwells = []
        for k in range(K):
            runs = np.diff(np.where(np.concatenate(
                [[True], s == k, [True]]))[0])
            if len(runs) > 0:
                dwells.append(runs.mean())
        row["dwell_mean_frames"] = float(np.mean(dwells)) if dwells else 0.0
        row["dwell_mean_s"] = row["dwell_mean_frames"] / FPS

        rows.append(row)

    return pd.DataFrame(rows)


def empirical_transition(df_states, K, group=None):
    if group is not None:
        sub = df_states[df_states["group"] == group]
    else:
        sub = df_states
    counts = np.zeros((K, K), dtype=float)
    for _, g in sub.groupby("video"):
        s = g.sort_values("frame")["hmm_state"].values
        if len(s) < 2:
            continue
        for a, b in zip(s[:-1], s[1:]):
            counts[a, b] += 1
    row = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(row > 0, counts / row, 0.0)
    return P


def _safe_mwu(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if np.allclose(a, a[0]) and np.allclose(b, b[0]) and a[0] == b[0]:
        return 1.0
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except ValueError:
        return np.nan


def cramers_v(contingency_table):
    """Cramer's V for a contingency table.

    Returns (V, chi2, p, dof, n). No bias correction.
    """
    chi2, p, dof, _ = chi2_contingency(contingency_table)
    n = contingency_table.values.sum()
    r, k = contingency_table.shape
    min_dim = min(r, k) - 1
    if min_dim <= 0 or n == 0:
        return np.nan, chi2, p, dof, n
    v = np.sqrt(chi2 / (n * min_dim))
    return float(v), float(chi2), float(p), int(dof), int(n)


# =========================================================
# 1. LOAD
# =========================================================

print("=" * 70)
print("LOAD")
print("=" * 70)

if not os.path.exists(FRAME_CSV):
    raise SystemExit(f"Not found: {FRAME_CSV}\n"
                     f"Run extract_frame_features.py first.")

df = pd.read_csv(FRAME_CSV, sep=";", decimal=",")
df = parse_meta(df)

print(f"Frames total: {len(df)}")
print(f"  by group:     {df['group'].value_counts().to_dict()}")
print(f"  by mouse:     {df['mouse_parsed'].value_counts().to_dict()}")
print(f"  by condition: {df['condition'].value_counts().to_dict()}")

# Show which videos fell into which group
print("\nGroups by video prefix:")
for g, sub in df.groupby("group"):
    vids = sorted(sub["video"].unique())
    print(f"  {g}: {len(vids)} videos")
    for v in vids[:5]:
        print(f"      {v}")
    if len(vids) > 5:
        print(f"      ... and {len(vids) - 5} more")

# Sanity check: both groups must be present
unique_groups = [g for g in df["group"].unique() if g != "unknown"]
if len(unique_groups) < 2:
    print("\n[!] WARNING: only one group found "
          f"({unique_groups}). blind vs sighted comparison will be skipped.")
    print("    Check GROUP_BY_PREFIX in CONFIG.\n")

# Sanity check: no NaN in mouse_parsed
n_nan_mouse = df["mouse_parsed"].isna().sum()
if n_nan_mouse > 0:
    print(f"[!] mouse_parsed contains {n_nan_mouse} NaN — these rows "
          f"will be lost during groupby. Check video names.")

missing = [c for c in HMM_FEATURES if c not in df.columns]
if missing:
    print(f"  [warn] missing features, dropping: {missing}")
    HMM_FEATURES = [c for c in HMM_FEATURES if c in df.columns]
    if len(HMM_FEATURES) < 3:
        raise SystemExit("Too few HMM features present.")

before = len(df)
df = df.dropna(subset=HMM_FEATURES + ["mouse_parsed", "video"]).reset_index(drop=True)
print(f"After dropna: {len(df)} (dropped {before - len(df)})")

if len(df) < 200:
    raise SystemExit("Too few frames for HMM. Check extractor output.")


# =========================================================
# 2. Z-SCORE PER MOUSE
# =========================================================

print("\n" + "=" * 70)
print("Z-SCORE PER MOUSE")
print("=" * 70)

df[HMM_FEATURES] = (df
                    .groupby("mouse_parsed")[HMM_FEATURES]
                    .transform(lambda s: (s - s.mean()) / s.std()))
df[HMM_FEATURES] = df[HMM_FEATURES].clip(-CLIP_Z, CLIP_Z)

print(f"Range after clip: "
      f"[{df[HMM_FEATURES].min().min():.2f}, "
      f"{df[HMM_FEATURES].max().max():.2f}]")


# =========================================================
# 3. BUILD SEQUENCES
# =========================================================

print("\n" + "=" * 70)
print("BUILD SEQUENCES")
print("=" * 70)

X_all, lengths, seq_meta = [], [], []

for (mouse, video), grp in df.groupby(
        ["mouse_parsed", "video"], sort=False):
    grp = grp.sort_values("frame")
    X_all.append(grp[HMM_FEATURES].values)
    lengths.append(len(grp))
    seq_meta.append({
        "mouse":     mouse,
        "video":     video,
        "group":     grp["group"].iloc[0],
        "condition": grp["condition"].iloc[0],
        "stim_dir":  grp["stim_dir"].iloc[0],
        "n_frames":  len(grp),
    })

X_all    = np.vstack(X_all)
lengths  = np.array(lengths)
seq_meta = pd.DataFrame(seq_meta)

# Sanity check: nothing lost
if lengths.sum() != len(df):
    raise SystemExit(
        f"[!] Sequences cover {lengths.sum()} of {len(df)} frames. "
        f"Some videos were dropped during groupby. "
        f"Check mouse_parsed / video for NaN.")

print(f"Sequences: {len(lengths)} videos, {len(X_all)} frames total")
print(f"Frames per video: min={lengths.min()}, "
      f"median={int(np.median(lengths))}, max={lengths.max()}")
print("Videos by group:")
print(seq_meta.groupby("group")["video"].count().to_string())


# =========================================================
# 4. MODEL SELECTION
# =========================================================

print("\n" + "=" * 70)
print(f"MODEL SELECTION (K = {K_RANGE})")
print("=" * 70)

results = []
for K in K_RANGE:
    model, ll = fit_best_model(X_all, lengths, K)
    if model is None:
        print(f"  K={K}  FAILED")
        continue

    D = X_all.shape[1]
    n_params = (K * D + K * D + K * (K - 1) + (K - 1))
    bic = -2 * ll + n_params * np.log(len(X_all))
    aic = -2 * ll + 2 * n_params
    results.append({
        "K": K, "log_likelihood": ll, "n_params": n_params,
        "BIC": bic, "AIC": aic,
    })
    print(f"  K={K}  LL={ll:10.2f}  BIC={bic:10.2f}  AIC={aic:10.2f}")

if not results:
    raise SystemExit("All K failed.")

results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_selection.csv"),
    sep=";", decimal=",", index=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(results_df["K"], results_df["BIC"], "o-", label="BIC")
ax.plot(results_df["K"], results_df["AIC"], "s--", label="AIC")
ax.set_xlabel("K (states)")
ax.set_ylabel("Information criterion")
ax.set_title("HMM model selection (frame-level)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "model_selection.png"), dpi=150)
plt.close(fig)


# =========================================================
# 5. FINAL FIT
# =========================================================

K_BIC   = int(results_df.loc[results_df["BIC"].idxmin(), "K"])
K_FINAL = min(K_BIC, K_MAX)
print(f"\nBest K by BIC = {K_BIC}  ->  using K = {K_FINAL}")

model, ll_final = fit_best_model(X_all, lengths, K_FINAL)
joblib.dump(model, os.path.join(OUTPUT_DIR, f"hmm_K{K_FINAL}.pkl"))


# =========================================================
# 6. EMISSIONS / TRANSITIONS / OCCUPANCY
# =========================================================

emissions = pd.DataFrame(
    model.means_,
    columns=HMM_FEATURES,
    index=[f"state_{k}" for k in range(K_FINAL)],
).round(3)

transitions = pd.DataFrame(
    model.transmat_,
    columns=[f"to_{k}"   for k in range(K_FINAL)],
    index  =[f"from_{k}" for k in range(K_FINAL)],
).round(3)

occupancy = pd.Series(
    model.startprob_,
    index=[f"state_{k}" for k in range(K_FINAL)],
    name="start_prob",
).round(3)

emissions.to_csv(os.path.join(OUTPUT_DIR, "hmm_emissions.csv"),
                 sep=";", decimal=",")
transitions.to_csv(os.path.join(OUTPUT_DIR, "hmm_transitions.csv"),
                   sep=";", decimal=",")
occupancy.to_csv(os.path.join(OUTPUT_DIR, "hmm_occupancy.csv"),
                 sep=";", decimal=",")

print("\nEMISSIONS (mean z-scored feature value per state):")
print(emissions.to_string())

print("\nTop-3 features per state (|mean|):")
for k in range(K_FINAL):
    row = emissions.iloc[k]
    top = row.abs().sort_values(ascending=False).head(3)
    print(f"  state_{k}: " + "  ".join(
        f"{f}={row[f]:+.2f}" for f in top.index))

fig, ax = plt.subplots(
    figsize=(1 + 0.5 * len(HMM_FEATURES), 1 + 0.6 * K_FINAL))
im = ax.imshow(model.means_, cmap="RdBu_r", aspect="auto",
               vmin=-2, vmax=2)
ax.set_xticks(range(len(HMM_FEATURES)))
ax.set_xticklabels(HMM_FEATURES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(K_FINAL))
ax.set_yticklabels([f"s{k}" for k in range(K_FINAL)])
ax.set_title("HMM emissions (z-scored, frame-level)")
for i in range(K_FINAL):
    for j in range(len(HMM_FEATURES)):
        ax.text(j, i, f"{model.means_[i, j]:.1f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(model.means_[i, j]) > 1.2 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "emissions_heatmap.png"), dpi=150)
plt.close(fig)


# =========================================================
# 7. VITERBI PER FRAME
# =========================================================

state_seqs = []
idx = 0
for L in lengths:
    _, s = model.decode(X_all[idx:idx + L], algorithm="viterbi")
    state_seqs.append(s)
    idx += L

df_states = df.copy()
df_states["hmm_state"] = np.concatenate(state_seqs)

df_states.to_csv(
    os.path.join(OUTPUT_DIR, "frame_states.csv"),
    sep=";", decimal=",", index=False)

print(f"\nFrame states distribution:")
print(df_states["hmm_state"].value_counts().sort_index().to_string())


# =========================================================
# 8. AGGREGATE (windows + videos)
# =========================================================

win_states = aggregate_window_states(df_states, K_FINAL)
win_states.to_csv(
    os.path.join(OUTPUT_DIR, "window_state_features.csv"),
    sep=";", decimal=",", index=False)

video_states = aggregate_video_states(df_states, K_FINAL)
video_states.to_csv(
    os.path.join(OUTPUT_DIR, "video_state_features.csv"),
    sep=";", decimal=",", index=False)


# =========================================================
# 9. STATE x CONDITION
# =========================================================

print("\n" + "=" * 70)
print("STATE x CONDITION")
print("=" * 70)

ct_raw  = pd.crosstab(df_states["hmm_state"], df_states["condition"])
ct_norm = pd.crosstab(df_states["hmm_state"], df_states["condition"],
                      normalize="columns").round(3)
print("Column-normalized (baseline vs stim):")
print(ct_norm.to_string())

if ct_raw.shape[1] > 1:
    v, chi2, p, dof, n = cramers_v(ct_raw)
    if np.isnan(v):
        strength = "n/a"
    elif v < 0.10:
        strength = "negligible"
    elif v < 0.30:
        strength = "weak"
    elif v < 0.50:
        strength = "moderate"
    else:
        strength = "strong"

    print(f"chi2={chi2:.1f}  p={p:.2e}  dof={dof}  N={n}")
    print(f"Cramer's V = {v:.4f}  ({strength} association)")


# =========================================================
# 10. GROUP COMPARISON (blind vs sighted) + PNG PLOTS
# =========================================================

print("\n" + "=" * 70)
print("GROUP COMPARISON - blind vs sighted")
print("=" * 70)

frac_cols = [f"frac_state_{k}" for k in range(K_FINAL)]
groups    = sorted([g for g in video_states["group"].unique()
                    if g != "unknown"])

summary_rows = []

if len(groups) < 2:
    print(f"  Only one group present: {groups}. "
          "Skipping comparison plots.")
else:
    g1, g2 = groups[0], groups[1]
    print(f"  Groups: {g1}  vs  {g2}")

    # 10.1. Per-feature statistics (Mann-Whitney)
    print("\n  Per-video fractions & stability:")
    for col in frac_cols + ["n_switches", "dwell_mean_s"]:
        a = video_states.loc[video_states["group"] == g1, col].dropna().values
        b = video_states.loc[video_states["group"] == g2, col].dropna().values
        p = _safe_mwu(a, b)
        flag = " *" if (not np.isnan(p) and p < 0.05) else ""
        if col in ("n_switches", "dwell_mean_s"):
            m1, m2 = np.median(a), np.median(b)
        else:
            m1, m2 = np.mean(a), np.mean(b)
        print(f"    {col:16s}  {g1}={m1:+.3f}  {g2}={m2:+.3f}  p={p:.4f}{flag}")
        summary_rows.append({
            "feature": col,
            "mean_" + g1: float(np.mean(a)),
            "mean_" + g2: float(np.mean(b)),
            "p_mannwhitney": float(p) if not np.isnan(p) else np.nan,
        })

    # 10.2. Per-frame occupancy
    occ = (df_states
           .groupby(["group", "hmm_state"])
           .size()
           .unstack(fill_value=0))
    occ_norm = occ.div(occ.sum(axis=1), axis=0)
    print("\n  Frame occupancy per state:")
    print(occ_norm.round(3).to_string())
    v_gs, chi2_gs, p_gs, dof_gs, n_gs = cramers_v(occ)
    print(f"\n  group x state association:")
    print(f"    chi2={chi2_gs:.1f}  p={p_gs:.2e}  dof={dof_gs}  N={n_gs}")
    print(f"    Cramer's V = {v_gs:.4f}")

    # PNG 1: bar plot of mean fractions per state
    fig, ax = plt.subplots(figsize=(1.2 * K_FINAL + 2, 4))
    x = np.arange(K_FINAL)
    w = 0.35
    a_means = [video_states.loc[video_states["group"] == g1,
                                f"frac_state_{k}"].mean()
               for k in range(K_FINAL)]
    b_means = [video_states.loc[video_states["group"] == g2,
                                f"frac_state_{k}"].mean()
               for k in range(K_FINAL)]
    ax.bar(x - w/2, a_means, w, label=g1)
    ax.bar(x + w/2, b_means, w, label=g2)
    ax.set_xticks(x); ax.set_xticklabels([f"s{k}" for k in range(K_FINAL)])
    ax.set_ylabel("mean fraction of frames")
    ax.set_title("State occupancy per group (per-video mean)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "group_occupancy_bars.png"),
                dpi=150)
    plt.close(fig)

    # PNG 2: boxplots of fractions per state
    ncols = min(4, K_FINAL)
    nrows = int(np.ceil(K_FINAL / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    for k in range(K_FINAL):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        a = video_states.loc[video_states["group"] == g1,
                             f"frac_state_{k}"].dropna().values
        b = video_states.loc[video_states["group"] == g2,
                             f"frac_state_{k}"].dropna().values
        ax.boxplot([a, b], labels=[g1, g2], showmeans=True)
        p = _safe_mwu(a, b)
        title = f"state_{k}"
        if not np.isnan(p):
            title += f"  p={p:.3f}" + (" *" if p < 0.05 else "")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    for k in range(K_FINAL, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    fig.suptitle("Per-video state fractions - blind vs sighted", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUTPUT_DIR, "group_fraction_boxplots.png"),
                dpi=150)
    plt.close(fig)

    # PNG 3: dwell time and number of switches
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, col, title in [
        (axes[0], "dwell_mean_s", "Mean dwell time per state (s)"),
        (axes[1], "n_switches",   "Number of state switches per video"),
    ]:
        a = video_states.loc[video_states["group"] == g1, col].dropna().values
        b = video_states.loc[video_states["group"] == g2, col].dropna().values
        ax.boxplot([a, b], labels=[g1, g2], showmeans=True)
        p = _safe_mwu(a, b)
        t = title + (f"  (p={p:.3f})" if not np.isnan(p) else "")
        ax.set_title(t, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "group_dwell_switches.png"),
                dpi=150)
    plt.close(fig)

    # PNG 4: mouse x state heatmap
    mouse_state = (df_states
                   .groupby(["mouse_parsed", "group", "hmm_state"])
                   .size()
                   .unstack(fill_value=0))
    mouse_state_norm = mouse_state.div(mouse_state.sum(axis=1), axis=0)
    mouse_state_norm = mouse_state_norm.reset_index()
    mouse_state_norm["group_rank"] = mouse_state_norm["group"].map(
        {g1: 0, g2: 1}).fillna(2)
    mouse_state_norm = mouse_state_norm.sort_values(
        ["group_rank", "mouse_parsed"]).set_index(
        ["mouse_parsed", "group"]).drop(columns="group_rank")
    mat = mouse_state_norm.values
    fig, ax = plt.subplots(
        figsize=(1.0 * K_FINAL + 2,
                 0.35 * len(mouse_state_norm) + 1.5))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(K_FINAL))
    ax.set_xticklabels([f"s{k}" for k in range(K_FINAL)], fontsize=8)
    ax.set_yticks(range(len(mouse_state_norm)))
    ax.set_yticklabels(
        [f"{m} [{g}]" for m, g in mouse_state_norm.index], fontsize=8)
    ax.set_title("Per-mouse state occupancy (fraction of frames)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    n_first = (mouse_state_norm.index.get_level_values("group") == g1).sum()
    ax.axhline(n_first - 0.5, color="white", lw=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "group_per_mouse_heatmap.png"),
                dpi=150)
    plt.close(fig)

    # PNG 5: empirical transition matrices
    P1 = empirical_transition(df_states, K_FINAL, group=g1)
    P2 = empirical_transition(df_states, K_FINAL, group=g2)
    fig, axes = plt.subplots(1, 3, figsize=(4.0 * K_FINAL / 2 + 6, 4.0))
    for ax, P, title in [(axes[0], P1, f"transitions: {g1}"),
                         (axes[1], P2, f"transitions: {g2}"),
                         (axes[2], P2 - P1, f"diff: {g2} - {g1}")]:
        vmax = 1.0 if title.startswith("transitions") else 0.5
        cmap = "Blues" if title.startswith("transitions") else "RdBu_r"
        im = ax.imshow(P, cmap=cmap, vmin=(-vmax if cmap == "RdBu_r" else 0),
                       vmax=vmax, aspect="auto")
        ax.set_xticks(range(K_FINAL))
        ax.set_xticklabels([f"->s{k}" for k in range(K_FINAL)],
                           rotation=45, fontsize=7)
        ax.set_yticks(range(K_FINAL))
        ax.set_yticklabels([f"s{k}->" for k in range(K_FINAL)], fontsize=7)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "group_transition_matrices.png"),
                dpi=150)
    plt.close(fig)

    # PNG 5b: transition difference only
    fig, ax = plt.subplots(figsize=(0.6 * K_FINAL + 2, 0.6 * K_FINAL + 2))
    im = ax.imshow(P2 - P1, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                   aspect="auto")
    for i in range(K_FINAL):
        for j in range(K_FINAL):
            ax.text(j, i, f"{P2[i,j]-P1[i,j]:+.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(P2[i,j]-P1[i,j]) > 0.3 else "black")
    ax.set_xticks(range(K_FINAL))
    ax.set_xticklabels([f"->s{k}" for k in range(K_FINAL)])
    ax.set_yticks(range(K_FINAL))
    ax.set_yticklabels([f"s{k}->" for k in range(K_FINAL)])
    ax.set_title(f"Delta transition probability: {g2} - {g1}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "group_transition_diff.png"),
                dpi=150)
    plt.close(fig)

    # PNG 6: representative timelines per group
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
    for ax, grp_name in zip(axes, [g1, g2]):
        sub_grp = df_states[df_states["group"] == grp_name]
        if len(sub_grp) == 0:
            ax.axis("off"); continue
        v = sub_grp["video"].iloc[0]
        v_sub = sub_grp[sub_grp["video"] == v].sort_values("frame")
        x = v_sub["time_s"] if "time_s" in v_sub.columns \
            else np.arange(len(v_sub)) / FPS
        ax.plot(x, v_sub["hmm_state"], ".-", ms=2)
        ax.set_ylim(-0.5, K_FINAL - 0.5)
        ax.set_yticks(range(K_FINAL))
        ax.set_ylabel("state")
        ax.set_title(f"{grp_name}: {v[:70]}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Representative state timelines by group", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUTPUT_DIR, "group_timelines.png"), dpi=130)
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "group_summary_stats.csv"),
        sep=";", decimal=",", index=False)

    print(f"\n  Saved PNG files:")
    for f in ["group_occupancy_bars.png",
              "group_fraction_boxplots.png",
              "group_dwell_switches.png",
              "group_per_mouse_heatmap.png",
              "group_transition_matrices.png",
              "group_transition_diff.png",
              "group_timelines.png"]:
        print(f"    {f}")


# =========================================================
# 11. TIMELINE PLOTS PER MOUSE
# =========================================================

print("\n" + "=" * 70)
print("TIMELINE PLOTS PER MOUSE")
print("=" * 70)

for mouse in df_states["mouse_parsed"].unique()[:4]:
    sub = df_states[df_states["mouse_parsed"] == mouse]
    video = sub["video"].iloc[0]
    sub = sub[sub["video"] == video].sort_values("frame")

    x = sub["time_s"] if "time_s" in sub.columns \
        else np.arange(len(sub)) / FPS
    feat_a = HMM_FEATURES[0]
    feat_b = HMM_FEATURES[-1]

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(x, sub["hmm_state"], ".-", ms=2)
    axes[0].set_ylabel("HMM state")
    axes[0].set_yticks(range(K_FINAL))
    axes[1].plot(x, sub[feat_a], lw=0.8)
    axes[1].set_ylabel(feat_a)
    axes[2].plot(x, sub[feat_b], lw=0.8)
    axes[2].set_ylabel(feat_b)
    axes[2].set_xlabel("time (s)")
    fig.suptitle(f"{mouse} - {video[:60]}", fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR,
                       f"timeline_{mouse}_{video[:30]}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved: {os.path.basename(out)}")


# =========================================================
# 12. PER-MOUSE CONSISTENCY
# =========================================================

print("\n" + "=" * 70)
print("PER-MOUSE STATE OCCUPANCY (fraction of frames)")
print("=" * 70)

per_mouse = (df_states
             .groupby(["mouse_parsed", "hmm_state"])
             .size()
             .unstack(fill_value=0))
per_mouse_norm = per_mouse.div(per_mouse.sum(axis=1), axis=0).round(3)
print(per_mouse_norm.to_string())


# =========================================================
# DONE
# =========================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"All results in: {OUTPUT_DIR}")