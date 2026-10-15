# analyze_head_turning_v3.py
# -*- coding: utf-8 -*-
"""
v3: исправлены merge (dedup), stim_dir (парсинг из video), PSD (порог длины).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import mannwhitneyu

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\90108\OneDrive\Рабочий стол\v\agouty"

FRAME_FEATURES_CSV = os.path.join(
    BASE_DIR, "frame_features10from10", "frame_features10.csv")
FRAME_STATES_CSV = os.path.join(
    BASE_DIR, "hmm_results_frames10of10(5)", "frame_states.csv")
OUTPUT_DIR = os.path.join(
    BASE_DIR, "hmm_results_frames10of10(5)", "head_turning_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS = 30
DRUM_SPEED_DEG_PER_S = 12.0
ANGULAR_CUTOFFS = [5, 8, 12, 18, 25, 40, 60]
OMR_BAND = (0.1, 0.8)
HEAD_SCANNING_STATE = 3
MIN_FRAMES_PSD = 128   # ~4.3 сек при 30 fps
STIM_SIGN = {"left": +1, "right": -1}   # ← поменяй, если знаки обратные

# ============================================================
# ПАРСИНГ stim_dir ИЗ ИМЕНИ ВИДЕО
# ============================================================

def parse_stim_dir(video: str) -> str:
    """Извлекает 'left'/'right' из имени видео.
    Работает для имён типа ..._stim001_..._left_... или ..._right_..."""
    if not isinstance(video, str):
        return ""
    name = video.lower()
    if re.search(r"_left(_|$|\.)", name) or "_left_" in name or \
       name.endswith("_left") or "_left." in name:
        return "left"
    if re.search(r"_right(_|$|\.)", name) or "_right_" in name or \
       name.endswith("_right") or "_right." in name:
        return "right"
    return ""

# ============================================================
# LOAD + MERGE (с dedup)
# ============================================================

print("=" * 70)
print("LOAD + MERGE (v3)")
print("=" * 70)

df_raw = pd.read_csv(FRAME_FEATURES_CSV, sep=";", decimal=",")
df_states = pd.read_csv(FRAME_STATES_CSV, sep=";", decimal=",")

print(f"Сырые: {len(df_raw)} строк")
print(f"Состояния: {len(df_states)} строк")

# --- ДИАГНОСТИКА дубликатов ---
raw_dup = df_raw.duplicated(subset=["video", "frame"], keep=False).sum()
st_dup = df_states.duplicated(subset=["video", "frame"], keep=False).sum()
print(f"\n[diag] дубликатов (video, frame):")
print(f"  в сыром файле:     {raw_dup}")
print(f"  в файле состояний: {st_dup}")

if raw_dup > 0:
    print("  → drop_duplicates по (video, frame) в сыром")
    before = len(df_raw)
    df_raw = df_raw.drop_duplicates(subset=["video", "frame"], keep="first")
    print(f"     {before} → {len(df_raw)}")

if st_dup > 0:
    print("  → drop_duplicates по (video, frame) в файле состояний")
    before = len(df_states)
    df_states = df_states.drop_duplicates(subset=["video", "frame"],
                                          keep="first")
    print(f"     {before} → {len(df_states)}")

# --- Мёрж ---
# Из сырого файла гарантируем уникальность ключа ещё раз
df_raw = df_raw.drop_duplicates(subset=["video", "frame"], keep="first")

# Из состояний берём ТОЛЬКО (video, frame, hmm_state)
state_slim = df_states[["video", "frame", "hmm_state"]].drop_duplicates(
    subset=["video", "frame"])

df = df_raw.merge(state_slim, on=["video", "frame"], how="inner")
print(f"\nПосле merge: {len(df)} строк (ожидалось ≤ {len(df_states)})")

if len(df) > len(df_states) * 1.05:
    raise SystemExit(
        f"[!] Merge всё ещё даёт больше строк, чем states. "
        f"Что-то не так с ключами. Проверь dtypes (video, frame).")

# --- Парсим stim_dir ---
df["stim_dir"] = df["video"].apply(parse_stim_dir)
print(f"\nstim_dir после парсинга:")
print(df.groupby(["condition", "stim_dir"]).size().to_string())

# --- Проверки ---
required = ["video", "frame", "group", "condition", "stim_dir",
            "mouse_parsed", "hmm_state", "angular_speed", "hb_speed"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Отсутствуют колонки: {missing}")

groups = sorted([g for g in df["group"].unique() if g != "unknown"])
g1, g2 = groups[0], groups[1]
print(f"\nГруппы: {g1}  vs  {g2}")
print(f"angular_speed: "
      f"[{df['angular_speed'].min():.1f}, "
      f"{df['angular_speed'].max():.1f}]  град/сек")


# ============================================================
# 1. STATE_3
# ============================================================

print("\n" + "=" * 70)
print(f"1. STATE_{HEAD_SCANNING_STATE} (Head scanning)")
print("=" * 70)

frac_in_state3 = (df
                  .assign(is_state3=(df["hmm_state"] == HEAD_SCANNING_STATE)
                          .astype(int))
                  .groupby(["mouse_parsed", "video", "group"])
                  .agg(frac_state3=("is_state3", "mean"),
                       n_frames=("is_state3", "size"))
                  .reset_index())

a = frac_in_state3[frac_in_state3["group"] == g1]["frac_state3"].values
b = frac_in_state3[frac_in_state3["group"] == g2]["frac_state3"].values
p_frac = mannwhitneyu(a, b, alternative="two-sided").pvalue if \
         (len(a) > 1 and len(b) > 1) else np.nan

print(f"  Доля времени в state_{HEAD_SCANNING_STATE}:")
print(f"    {g1}: median={np.median(a):.3f}  n={len(a)}")
print(f"    {g2}: median={np.median(b):.3f}  n={len(b)}")
print(f"    Mann-Whitney p = {p_frac:.4f}")

frac_in_state3.to_csv(
    os.path.join(OUTPUT_DIR, "frac_state3_per_video.csv"),
    sep=";", decimal=",", index=False)

sub3 = df[df["hmm_state"] == HEAD_SCANNING_STATE].copy()
sub3["abs_angular_speed"] = sub3["angular_speed"].abs()

print(f"\n  |angular_speed| внутри state_{HEAD_SCANNING_STATE}:")
for g in [g1, g2]:
    x = sub3[sub3["group"] == g]["abs_angular_speed"].dropna().values
    print(f"    {g}: n={len(x)}  median={np.median(x):.1f}  "
          f"mean={x.mean():.1f}  p90={np.percentile(x, 90):.1f}")

xg1 = sub3[sub3["group"] == g1]["abs_angular_speed"].dropna().values
xg2 = sub3[sub3["group"] == g2]["abs_angular_speed"].dropna().values
p_ang = mannwhitneyu(xg1, xg2, alternative="two-sided").pvalue

fig, ax = plt.subplots(figsize=(8, 4))
for g, color in zip([g1, g2], ["steelblue", "darkorange"]):
    x = sub3[sub3["group"] == g]["abs_angular_speed"].dropna().values
    if len(x) > 0:
        ax.hist(x, bins=80, range=(0, 300), alpha=0.5, density=True,
                label=f"{g} (n={len(x)})", color=color)
ax.axvline(DRUM_SPEED_DEG_PER_S, color="red", ls="--",
           label=f"барабан = {DRUM_SPEED_DEG_PER_S}°/сек")
ax.set_xlabel("|angular_speed| (deg/s)")
ax.set_ylabel("density")
ax.set_title(f"state_{HEAD_SCANNING_STATE}: |angular_speed|  (p={p_ang:.2e})")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "state3_angular_speed_hist.png"),
            dpi=150)
plt.close(fig)


# ============================================================
# 2. ПОРОГОВЫЙ АНАЛИЗ
# ============================================================

print("\n" + "=" * 70)
print("2. Пороговый анализ |angular_speed|")
print("=" * 70)

df["abs_angular_speed"] = df["angular_speed"].abs()

for cutoff in ANGULAR_CUTOFFS:
    df[f"above_{cutoff}"] = (df["abs_angular_speed"] > cutoff).astype(int)

# --- Все видео
print("  Все видео:")
rows = []
for cutoff in ANGULAR_CUTOFFS:
    per_video = (df
                 .groupby(["mouse_parsed", "video", "group"])
                 .agg(frac_above=(f"above_{cutoff}", "mean"))
                 .reset_index())
    a = per_video[per_video["group"] == g1]["frac_above"].values
    b = per_video[per_video["group"] == g2]["frac_above"].values
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    print(f"    cutoff={cutoff:3d}°/s: {g1}={a.mean():.4f}  "
          f"{g2}={b.mean():.4f}  p={p:.4f}"
          + (" *" if p < 0.05 else ""))
    rows.append({"cutoff": cutoff, f"mean_{g1}": a.mean(),
                 f"mean_{g2}": b.mean(), "p": p})

pd.DataFrame(rows).to_csv(
    os.path.join(OUTPUT_DIR, "angular_threshold_analysis.csv"),
    sep=";", decimal=",", index=False)

# --- Только stim
print("\n  Только stim-видео:")
stim = df[df["condition"] == "stim"]
for cutoff in ANGULAR_CUTOFFS:
    per_video = (stim
                 .groupby(["mouse_parsed", "video", "group"])
                 .agg(frac_above=(f"above_{cutoff}", "mean"))
                 .reset_index())
    a = per_video[per_video["group"] == g1]["frac_above"].values
    b = per_video[per_video["group"] == g2]["frac_above"].values
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    print(f"    cutoff={cutoff:3d}: {g1}={a.mean():.4f}  "
          f"{g2}={b.mean():.4f}  p={p:.4f}" + (" *" if p < 0.05 else ""))

# --- Гистограмма на stim
fig, ax = plt.subplots(figsize=(9, 4))
for g, color in zip([g1, g2], ["steelblue", "darkorange"]):
    x = df[(df["group"] == g) & (df["condition"] == "stim")][
        "angular_speed"].dropna().values
    ax.hist(x, bins=150, range=(-400, 400), alpha=0.5, density=True,
            label=f"{g} stim (n={len(x)})", color=color)
ax.axvline(+DRUM_SPEED_DEG_PER_S, color="red", ls="--", lw=2,
           label=f"+{DRUM_SPEED_DEG_PER_S}°/сек")
ax.axvline(-DRUM_SPEED_DEG_PER_S, color="red", ls="--", lw=2,
           label=f"−{DRUM_SPEED_DEG_PER_S}°/сек")
ax.set_xlabel("angular_speed (°/s)")
ax.set_ylabel("density")
ax.set_title("Распределение angular_speed на stim-видео")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "angular_speed_stim_hist.png"),
            dpi=150)
plt.close(fig)


# ============================================================
# 3. Welch PSD (с понижённым порогом длины)
# ============================================================

print("\n" + "=" * 70)
print(f"3. Welch PSD, ОМР-полоса {OMR_BAND} Гц  "
      f"(min длина = {MIN_FRAMES_PSD} кадров)")
print("=" * 70)


def psd_features(x, fs, band, min_len):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < min_len:
        return np.nan, np.nan
    nperseg = min(128, len(x))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    in_band = (f >= band[0]) & (f <= band[1])
    band_power = float(Pxx[in_band].mean()) if in_band.any() else np.nan
    return band_power, float(Pxx.mean())


rows = []
for (mouse, video), g in df.groupby(["mouse_parsed", "video"], sort=False):
    g = g.sort_values("frame")
    band_power, total_power = psd_features(
        g["angular_speed"].values, FPS, OMR_BAND, MIN_FRAMES_PSD)
    rows.append({
        "mouse_parsed": mouse, "video": video,
        "group": g["group"].iloc[0],
        "condition": g["condition"].iloc[0],
        "stim_dir": g["stim_dir"].iloc[0],
        "n_frames": len(g),
        "omr_power": band_power,
        "total_power": total_power,
        "omr_ratio": (band_power / total_power
                      if (total_power and not np.isnan(total_power))
                      else np.nan),
    })

psd_df = pd.DataFrame(rows)
psd_df.to_csv(os.path.join(OUTPUT_DIR, "psd_features.csv"),
              sep=";", decimal=",", index=False)

stim_psd = psd_df[psd_df["condition"] == "stim"].dropna(
    subset=["omr_power", "omr_ratio"])
print(f"  Stim-видео с длиной ≥ {MIN_FRAMES_PSD}: n={len(stim_psd)} "
      f"({g1}: {(stim_psd['group']==g1).sum()}, "
      f"{g2}: {(stim_psd['group']==g2).sum()})")

for metric in ["omr_power", "omr_ratio"]:
    a = stim_psd[stim_psd["group"] == g1][metric].values
    b = stim_psd[stim_psd["group"] == g2][metric].values
    if len(a) < 2 or len(b) < 2:
        print(f"    {metric}: слишком мало данных (n={len(a)}, {len(b)})")
        continue
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    print(f"    {metric:12s}: {g1}={np.median(a):.4f}  "
          f"{g2}={np.median(b):.4f}  p={p:.4f}"
          + (" *" if p < 0.05 else ""))

if len(stim_psd) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["omr_power", "omr_ratio"]):
        a = stim_psd[stim_psd["group"] == g1][metric].dropna().values
        b = stim_psd[stim_psd["group"] == g2][metric].dropna().values
        if len(a) > 0 and len(b) > 0:
            ax.boxplot([a, b], tick_labels=[g1, g2], showmeans=True)
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue \
                if (len(a) > 1 and len(b) > 1) else np.nan
            ax.set_title(metric + (f"  p={p:.3f}"
                                   if not np.isnan(p) else ""))
            ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"ОМР-полоса {OMR_BAND} Гц, stim-видео")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUTPUT_DIR, "psd_omr_comparison.png"),
                dpi=150)
    plt.close(fig)


# ============================================================
# 4. ОМР-ТЕСТ: направление + скорость вокруг 12°/сек
# ============================================================

print("\n" + "=" * 70)
print("4. ОМР-тест (медленная фаза, 12°/сек)")
print("=" * 70)

# --- СНАЧАЛА проверь, что stim_dir распарсился ---
stim = df[df["condition"] == "stim"].copy()
stim["expected_sign"] = stim["stim_dir"].map(STIM_SIGN)

print(f"  stim_dir на stim-видео:")
print(f"    {stim['stim_dir'].value_counts().to_dict()}")
print(f"  Кадров с known direction: "
      f"{stim['expected_sign'].notna().sum()} / {len(stim)}")

if stim["expected_sign"].notna().sum() == 0:
    print("\n  [!] stim_dir не распарсился. Уникальные значения:")
    print(f"      {stim['video'].unique()[:5]}")
    print(f"      Проверь, есть ли в имени видео '_left_' или '_right_'.")
    raise SystemExit("Останов: без stim_dir ОМР-тест невозможен.")


# --- Узкое окно вокруг 12°/сек ---
OMR_LOW  = 8.0    # град/сек
OMR_HIGH = 16.0   # град/сек

def omr_fraction_per_video(g, low=OMR_LOW, high=OMR_HIGH):
    """Доля кадров, где голова движется в сторону барабана,
    при этом |angular_speed| попадает в окно вокруг 12°/сек."""
    x = g["angular_speed"].values
    exp = g["expected_sign"].values

    mask = (np.isfinite(x) & np.isfinite(exp)
            & (np.abs(x) >= low) & (np.abs(x) <= high))
    n_valid = int(mask.sum())
    if n_valid == 0:
        return np.nan, 0, 0

    same = (np.sign(x[mask]) == np.sign(exp[mask]))
    frac = float(same.mean())
    return frac, n_valid, int(same.sum())


rows = []
for (mouse, video), g in stim.groupby(
        ["mouse_parsed", "video"], sort=False):
    g = g.sort_values("frame")
    frac, n_valid, n_same = omr_fraction_per_video(g)
    rows.append({
        "mouse_parsed": mouse,
        "video": video,
        "group": g["group"].iloc[0],
        "stim_dir": g["stim_dir"].iloc[0],
        "n_frames": len(g),
        "n_valid_frames": n_valid,
        "n_same_direction": n_same,
        "omr_frac": frac,
    })

omr_df = pd.DataFrame(rows)
omr_df.to_csv(
    os.path.join(OUTPUT_DIR, "omr_direction_fraction.csv"),
    sep=";", decimal=",", index=False)

# --- Сводка ---
valid = omr_df.dropna(subset=["omr_frac"])
print(f"\n  Видео с ненулевым числом кадров в окне "
      f"[{OMR_LOW}, {OMR_HIGH}]°/с: {len(valid)}")
print(f"  Суммарно кадров в окне:")
print(f"    {g1}: {valid[valid['group']==g1]['n_valid_frames'].sum()}")
print(f"    {g2}: {valid[valid['group']==g2]['n_valid_frames'].sum()}")

if len(valid) > 0:
    a = valid[valid["group"] == g1]["omr_frac"].values
    b = valid[valid["group"] == g2]["omr_frac"].values
    print(f"\n  Доля кадров, где голова движется в сторону барабана:")
    print(f"    {g1}: median={np.median(a):.3f}  mean={a.mean():.3f}  n={len(a)}")
    print(f"    {g2}: median={np.median(b):.3f}  mean={b.mean():.3f}  n={len(b)}")

    if len(a) > 1 and len(b) > 1:
        p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        print(f"    Mann-Whitney p = {p:.4f}" + (" *" if p < 0.05 else ""))

    # --- Ключевой тест: отличается ли от 0.5 (случайное направление)? ---
    from scipy.stats import wilcoxon
    for g, vals in [(g1, a), (g2, b)]:
        if len(vals) > 5:
            try:
                stat, p_vs_half = wilcoxon(vals - 0.5)
                sig = " *" if p_vs_half < 0.05 else ""
                direction = "в сторону барабана" if np.median(vals) > 0.5 \
                            else "ПРОТИВ барабана"
                print(f"    {g}: vs 0.5 -> p={p_vs_half:.4f}{sig}  "
                      f"(median {'>' if np.median(vals)>0.5 else '<'} 0.5, "
                      f"{direction})")
            except Exception as e:
                print(f"    {g}: wilcoxon failed: {e}")

    # --- Boxplot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([a, b], tick_labels=[g1, g2], showmeans=True)
    ax.axhline(0.5, color="red", ls="--", lw=2,
               label="случайное направление (0.5)")
    ax.set_ylabel(f"Доля кадров со направлением ОМР\n"
                  f"(|v| в [{OMR_LOW}, {OMR_HIGH}]°/с)")
    ax.set_title(f"ОМР-направленность вокруг {DRUM_SPEED_DEG_PER_S}°/с")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "omr_direction_boxplot.png"),
                dpi=150)
    plt.close(fig)


# --- Разбивка по stim_dir (left/right отдельно) ---
print("\n  Разбивка по направлению стимула:")
for sd in ["left", "right"]:
    sub = valid[valid["stim_dir"] == sd]
    if len(sub) == 0:
        continue
    a = sub[sub["group"] == g1]["omr_frac"].values
    b = sub[sub["group"] == g2]["omr_frac"].values
    print(f"    stim_dir={sd}:")
    print(f"      {g1}: median={np.median(a):.3f}  n={len(a)}")
    print(f"      {g2}: median={np.median(b):.3f}  n={len(b)}")
    if len(a) > 1 and len(b) > 1:
        p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        print(f"      Mann-Whitney p = {p:.4f}" + (" *" if p < 0.05 else ""))