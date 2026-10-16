# omr_by_state.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\90108\OneDrive\Рабочий стол\v\agouty"
FRAME_FEATURES_CSV = os.path.join(
    BASE_DIR, "frame_features10", "frame_features10.csv")
FRAME_STATES_CSV = os.path.join(
    BASE_DIR, "hmm_results_frames10of10(5)", "frame_states.csv")
OUTPUT_DIR = os.path.join(
    BASE_DIR, "hmm_results_frames10of10(5)", "paper_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS = 30
DRUM_SPEED = 12.0
OMR_LOW, OMR_HIGH = 8.0, 16.0
STIM_SIGN = {"left": +1, "right": -1}

COL_BLIND   = "#4472C4"
COL_SIGHTED = "#ED7D31"

STATE_NAMES = {
    0: "s0 Head retracted",
    1: "s1 Resting",
    2: "s2 Locomotion, slight turn",
    3: "s3 Orienting",
    4: "s4 Neutral exploration",
}


# ============================================================
# HELPERS
# ============================================================

def parse_dir(video):
    if not isinstance(video, str):
        return ""
    vl = video.lower()
    if "_left_" in vl or vl.endswith("_left"):
        return "left"
    if "_right_" in vl or vl.endswith("_right"):
        return "right"
    return ""


# ============================================================
# LOAD + MERGE
# ============================================================

print("=" * 70)
print("LOAD + MERGE")
print("=" * 70)

df_raw = pd.read_csv(FRAME_FEATURES_CSV, sep=";", decimal=",")
df_states = pd.read_csv(FRAME_STATES_CSV, sep=";", decimal=",")

df_raw = df_raw.drop_duplicates(subset=["video", "frame"], keep="first")
df_states = df_states[["video", "frame", "hmm_state"]].drop_duplicates(
    subset=["video", "frame"])

df = df_raw.merge(df_states, on=["video", "frame"], how="inner")
df["stim_dir"] = df["video"].apply(parse_dir)
df["abs_v"] = df["angular_speed"].abs()
df["expected_sign"] = df["stim_dir"].map(STIM_SIGN)

groups = sorted([g for g in df["group"].unique() if g != "unknown"])
g1, g2 = groups[0], groups[1]
print(f"Всего строк: {len(df)}")
print(f"Группы: {g1}  vs  {g2}")
print(f"HMM-состояний: {sorted(df['hmm_state'].unique())}")


# ============================================================
# STIM + OMR WINDOW
# ============================================================

stim = df[(df["condition"] == "stim") & (df["stim_dir"] != "")].copy()
stim["in_window"] = (stim["abs_v"] >= OMR_LOW) & (stim["abs_v"] <= OMR_HIGH)

stim_win = stim[stim["in_window"]].copy()
stim_win["aligned"] = (
    np.sign(stim_win["angular_speed"]) == stim_win["expected_sign"])

print(f"\nВсего stim-кадров: {len(stim)}")
print(f"Кадров в окне [{OMR_LOW}, {OMR_HIGH}]°/с: {len(stim_win)}")

print("\nРаспределение ОМР-кадров по состояниям:")
by_state = stim_win.groupby(["hmm_state", "group"]).size().unstack(fill_value=0)
print(by_state.to_string())


# ============================================================
# OMR BY STATE
# ============================================================

print("\n" + "=" * 70)
print("ОМР-ТЕСТ РАЗДЕЛЬНО ПО HMM-СОСТОЯНИЯМ")
print("=" * 70)
print(f"{'state':>4} | {'name':<28} | "
      f"{g1:>18} | {g2:>18} | p-value")
print("-" * 95)

results_by_state = {}

for state in sorted(stim_win["hmm_state"].unique()):
    sub = stim_win[stim_win["hmm_state"] == state]

    per_video = []
    for (mouse, video), g in sub.groupby(
            ["mouse_parsed", "video"], sort=False):
        if len(g) < 5:
            continue
        per_video.append({
            "mouse_parsed": mouse,
            "video": video,
            "group": g["group"].iloc[0],
            "frac_in_drum_dir": float(g["aligned"].mean()),
            "n_in_state": len(g),
        })
    v = pd.DataFrame(per_video)

    a = v[v["group"] == g1]["frac_in_drum_dir"].values
    b = v[v["group"] == g2]["frac_in_drum_dir"].values

    state_name = STATE_NAMES.get(state, f"state_{state}")

    if len(a) < 3 or len(b) < 3:
        print(f"{state:>4} | {state_name:<28} | "
              f"n={len(a):>3} insufficient    | "
              f"n={len(b):>3} insufficient    | —")
        results_by_state[state] = {
            "name": state_name,
            f"n_{g1}": len(a), f"n_{g2}": len(b),
            f"median_{g1}": np.nan, f"median_{g2}": np.nan,
            "p_mannwhitney": np.nan,
        }
        continue

    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    flag = " *" if p < 0.05 else ""

    print(f"{state:>4} | {state_name:<28} | "
          f"med={np.median(a):>6.3f} (n={len(a):>3}) | "
          f"med={np.median(b):>6.3f} (n={len(b):>3}) | "
          f"p={p:.4f}{flag}")

    results_by_state[state] = {
        "name": state_name,
        f"n_{g1}": len(a),
        f"n_{g2}": len(b),
        f"median_{g1}": float(np.median(a)),
        f"median_{g2}": float(np.median(b)),
        "p_mannwhitney": float(p),
    }


# ============================================================
# VS 0.5 WITHIN EACH GROUP AND STATE
# ============================================================

print("\n" + "=" * 70)
print("ОТКЛОНЕНИЕ ОТ СЛУЧАЙНОГО (0.5) ВНУТРИ КАЖДОЙ ГРУППЫ")
print("=" * 70)

for state in sorted(stim_win["hmm_state"].unique()):
    sub = stim_win[stim_win["hmm_state"] == state]
    per_video = []
    for (mouse, video), g in sub.groupby(
            ["mouse_parsed", "video"], sort=False):
        if len(g) < 5:
            continue
        per_video.append({
            "group": g["group"].iloc[0],
            "frac": float(g["aligned"].mean()),
        })
    v = pd.DataFrame(per_video)
    a = v[v["group"] == g1]["frac"].values
    b = v[v["group"] == g2]["frac"].values
    state_name = STATE_NAMES.get(state, f"state_{state}")

    if len(a) < 5 or len(b) < 5:
        continue

    try:
        _, p_a = wilcoxon(a - 0.5)
        _, p_b = wilcoxon(b - 0.5)
        dir_a = ">" if np.median(a) > 0.5 else "<"
        dir_b = ">" if np.median(b) > 0.5 else "<"
        print(f"{state_name:<28} | "
              f"{g1}: median={np.median(a):.3f} ({dir_a}0.5) p={p_a:.4f}"
              + (" *" if p_a < 0.05 else "") + " | "
              f"{g2}: median={np.median(b):.3f} ({dir_b}0.5) p={p_b:.4f}"
              + (" *" if p_b < 0.05 else ""))
    except Exception as e:
        print(f"{state_name:<28} | wilcoxon failed: {e}")


# ============================================================
# SAVE TABLE
# ============================================================

summary = pd.DataFrame(results_by_state).T
summary.index.name = "hmm_state"
summary.to_csv(
    os.path.join(OUTPUT_DIR, "omr_by_state.csv"),
    sep=";", decimal=",")
print(f"\n→ omr_by_state.csv")


# ============================================================
# FIGURE — BAR PLOT BY STATE
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))

states = sorted(results_by_state.keys())
x = np.arange(len(states))
w = 0.38

a_medians = [results_by_state[s].get(f"median_{g1}", np.nan) for s in states]
b_medians = [results_by_state[s].get(f"median_{g2}", np.nan) for s in states]
p_vals = [results_by_state[s].get("p_mannwhitney", np.nan) for s in states]

bars1 = ax.bar(x - w/2, a_medians, w, label=g1,
               color=COL_BLIND, edgecolor="black", lw=0.6, alpha=0.85)
bars2 = ax.bar(x + w/2, b_medians, w, label=g2,
               color=COL_SIGHTED, edgecolor="black", lw=0.6, alpha=0.85)

for i, p in enumerate(p_vals):
    if not np.isnan(p):
        y_top = max(a_medians[i] if not np.isnan(a_medians[i]) else 0,
                    b_medians[i] if not np.isnan(b_medians[i]) else 0)
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(i, y_top + 0.03, stars, ha="center", fontsize=11)

ax.axhline(0.5, color="red", ls="--", lw=1.5,
           label="Случайное (0.5)", zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([STATE_NAMES.get(s, f"s{s}") for s in states],
                   rotation=15, ha="right", fontsize=9)
ax.set_ylabel("ОМР-индекс (median по видео)")
ax.set_title("ОМР-индекс в разных HMM-состояниях\n"
             "(median по видео, планки — медиана, "
             "звёздочки — Mann–Whitney)", fontsize=11)
ax.set_ylim(0, 1)
ax.legend(loc="upper right")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "figure_OMR_by_state.png"),
            dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_DIR, "figure_OMR_by_state.pdf"),
            bbox_inches="tight")
plt.close(fig)
print(f"→ figure_OMR_by_state.png / .pdf")


# ============================================================
# FIGURE — BOXPLOTS BY STATE
# ============================================================

fig, axes = plt.subplots(1, len(states),
                         figsize=(3.2 * len(states), 4.5),
                         sharey=True)

for ax, state in zip(axes, states):
    sub = stim_win[stim_win["hmm_state"] == state]
    per_video = []
    for (mouse, video), g in sub.groupby(
            ["mouse_parsed", "video"], sort=False):
        if len(g) < 5:
            continue
        per_video.append({
            "group": g["group"].iloc[0],
            "frac": float(g["aligned"].mean()),
        })
    v = pd.DataFrame(per_video)
    a = v[v["group"] == g1]["frac"].values
    b = v[v["group"] == g2]["frac"].values

    if len(a) < 3 or len(b) < 3:
        ax.axis("off")
        continue

    bp = ax.boxplot([a, b], positions=[0, 1], widths=0.6,
                    patch_artist=True, showmeans=False,
                    medianprops=dict(color="black", lw=1.5))
    bp["boxes"][0].set_facecolor(COL_BLIND);   bp["boxes"][0].set_alpha(0.35)
    bp["boxes"][1].set_facecolor(COL_SIGHTED); bp["boxes"][1].set_alpha(0.35)

    rng = np.random.default_rng(0)
    for pos, data, col in zip([0, 1], [a, b], [COL_BLIND, COL_SIGHTED]):
        jitter = rng.uniform(-0.1, 0.1, size=len(data))
        ax.scatter(pos + jitter, data, s=25, color=col, alpha=0.7,
                   edgecolor="black", lw=0.3, zorder=3)

    ax.axhline(0.5, color="red", ls="--", lw=1.2)
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.set_title(f"{STATE_NAMES.get(state, f's{state}')}\np={p:.3f} {stars}",
                 fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g1, g2], fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, 1)

axes[0].set_ylabel("ОМР-индекс")
fig.suptitle("ОМР-индекс в каждом HMM-состоянии: "
             "распределение по видео", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUTPUT_DIR, "figure_OMR_by_state_boxplots.png"),
            dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_DIR, "figure_OMR_by_state_boxplots.pdf"),
            bbox_inches="tight")
plt.close(fig)
print(f"→ figure_OMR_by_state_boxplots.png / .pdf")


print("\n" + "=" * 70)
print("ГОТОВО")
print("=" * 70)
print(f"Результаты в: {OUTPUT_DIR}")