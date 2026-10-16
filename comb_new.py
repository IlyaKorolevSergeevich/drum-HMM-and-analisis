# extract_frame_features_v2.py
# -*- coding: utf-8 -*-
"""
Frame-level feature extraction (v2).
Одна строка = один кадр. Углы в sin/cos, признаки нормированы,
скорости считаются по сглаженным координатам.

Выход идёт в HMM (hmm_train_frames.py с обновлённым HMM_FEATURES).
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ---------- CONFIG ----------
FILE       = r"C:\Users\90108\OneDrive\Рабочий стол\c3h\5\UNKNOWN_c3h5_combined.csv"
OUTPUT_DIR = r"C:\Users\90108\OneDrive\Рабочий стол\v\agouty\frame_features10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_NAME = "frame_features_c3h5_v2.csv"

FPS                  = 30
LIKELIHOOD_THRESHOLD = 0.70
SMOOTH_WINDOW        = 5      # для финальных признаков
POS_SMOOTH_WINDOW    = 7      # для координат перед скоростями

REQUIRED_BODYPARTS = [
    "left_ear_base", "right_ear_base", "head_base", "tail_base",
]

# Группа определяется по префиксу имени видео
GROUP_BY_PREFIX = {"c3h": "blind", "cba": "sighted"}

EPS = 1e-6


# ---------- HELPERS ----------
def wrap_angle(a):
    """Приводит угол к [-180, 180)."""
    return (a + 180) % 360 - 180


def smooth(x, w):
    return (pd.Series(x)
            .rolling(w, center=True, min_periods=1)
            .mean().values)


# ---------- MAIN ----------
print("=" * 70)
print("FILE:", FILE)
print("=" * 70)

df = pd.read_csv(FILE, sep=";", decimal=",")

for col in df.columns:
    if col == "video":
        continue
    if df[col].dtype == object:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

all_frames = []

for video in df["video"].unique():
    d = df[df["video"] == video].copy().reset_index(drop=True)
    n_raw = len(d)

    # --- Интерполяция точек с низкой уверенностью ---
    for bp in REQUIRED_BODYPARTS:
        for ax in ["x", "y"]:
            col = f"{bp}_{ax}"
            lk_col = f"{bp}_likelihood"
            if col in d.columns and lk_col in d.columns:
                bad = d[lk_col] < LIKELIHOOD_THRESHOLD
                d.loc[bad, col] = np.nan
                d[col] = d[col].interpolate(limit_direction="both")

    # --- mean_likelihood как признак качества ---
    lk_cols = [f"{bp}_likelihood" for bp in REQUIRED_BODYPARTS
               if f"{bp}_likelihood" in d.columns]
    if lk_cols:
        d["mean_likelihood"] = d[lk_cols].mean(axis=1)
    else:
        d["mean_likelihood"] = np.nan

    # --- Фильтр кадров: >= 3 из 4 точек надёжны ---
    valid = sum(
        (d[f"{bp}_likelihood"] > LIKELIHOOD_THRESHOLD).astype(int)
        for bp in REQUIRED_BODYPARTS
        if f"{bp}_likelihood" in d.columns
    )
    d = d[valid >= 3].reset_index(drop=True)
    n_frames = len(d)
    if n_frames < 30:
        print(f"  skip {video[:55]}: only {n_frames} valid frames")
        continue

    # --- Сглаживаем координаты ДО скоростей ---
    Lx = smooth(d["left_ear_base_x"].values,  POS_SMOOTH_WINDOW)
    Ly = smooth(d["left_ear_base_y"].values,  POS_SMOOTH_WINDOW)
    Rx = smooth(d["right_ear_base_x"].values, POS_SMOOTH_WINDOW)
    Ry = smooth(d["right_ear_base_y"].values, POS_SMOOTH_WINDOW)
    Hx = smooth(d["head_base_x"].values,      POS_SMOOTH_WINDOW)
    Hy = smooth(d["head_base_y"].values,      POS_SMOOTH_WINDOW)
    Tx = smooth(d["tail_base_x"].values,      POS_SMOOTH_WINDOW)
    Ty = smooth(d["tail_base_y"].values,      POS_SMOOTH_WINDOW)

    # --- Геометрия головы ---
    ear_mid_x = (Lx + Rx) / 2.0
    ear_mid_y = (Ly + Ry) / 2.0

    # направление головы = перпендикуляр к линии ушей
    ear_dx = Rx - Lx
    ear_dy = Ry - Ly
    head_dx = -ear_dy
    head_dy =  ear_dx

    angle_raw = np.degrees(np.arctan2(head_dy, head_dx))

    # направление тела: от tail_base к head_base
    body_angle = np.degrees(np.arctan2(Hy - Ty, Hx - Tx))

    # выбираем из angle_raw и angle_raw+180 то, что ближе к телу
    diff1 = np.abs(wrap_angle(angle_raw       - body_angle))
    diff2 = np.abs(wrap_angle(angle_raw + 180 - body_angle))
    head_angle = np.where(diff1 < diff2,
                          angle_raw,
                          wrap_angle(angle_raw + 180))

    # --- Углы как sin/cos (устраняет проблему wrap) ---
    ha_rad          = np.radians(head_angle)
    head_angle_sin  = np.sin(ha_rad)
    head_angle_cos  = np.cos(ha_rad)

    hbd             = wrap_angle(head_angle - body_angle)   # относит. угол
    hbd_rad         = np.radians(hbd)
    head_body_diff_sin = np.sin(hbd_rad)
    head_body_diff_cos = np.cos(hbd_rad)

    # --- Метрики размера (нормированные) ---
    head_area = 0.5 * np.abs(
        (Lx - Hx) * (Ry - Hy) - (Rx - Hx) * (Ly - Hy)
    )
    ear_line = np.sqrt((Rx - Lx) ** 2 + (Ry - Ly) ** 2) + EPS
    head_len = np.sqrt((ear_mid_x - Hx) ** 2 + (ear_mid_y - Hy) ** 2) + EPS

    # нормировки: убирают зависимость от размера кадра и масштаба
    head_area_norm = head_area / (ear_line * head_len)
    head_len_norm  = head_len / ear_line

    d_L = np.sqrt((Lx - Hx) ** 2 + (Ly - Hy) ** 2)
    d_R = np.sqrt((Rx - Hx) ** 2 + (Ry - Hy) ** 2)
    ear_asym_norm = (d_L - d_R) / (d_L + d_R + EPS)   # в [-1, 1]

    # --- Движение ---
    hb_vx = np.gradient(Hx) * FPS
    hb_vy = np.gradient(Hy) * FPS
    hb_speed = np.sqrt(hb_vx ** 2 + hb_vy ** 2)

    # угловая скорость: круговая разность head_angle между кадрами
    d_angle = np.diff(np.concatenate([[head_angle[0]], head_angle]))
    d_angle = wrap_angle(d_angle)
    angular_speed = smooth(d_angle * FPS, SMOOTH_WINDOW)

    # --- Финальное сглаживание признаков ---
    def sm(x):
        return smooth(x, SMOOTH_WINDOW)

    out = pd.DataFrame({
        "video":               video,
        "frame":               np.arange(n_frames),
        "time_s":              np.arange(n_frames) / FPS,

        # Pose
        "head_area_norm":      sm(head_area_norm),
        "head_len_norm":       sm(head_len_norm),
        "ear_asym_norm":       sm(ear_asym_norm),

        # Orientation (круговая, sin/cos)
        "head_angle_sin":      sm(head_angle_sin),
        "head_angle_cos":      sm(head_angle_cos),
        "head_body_diff_sin":  sm(head_body_diff_sin),
        "head_body_diff_cos":  sm(head_body_diff_cos),

        # Movement
        "hb_speed":            sm(hb_speed),
        "angular_speed":       angular_speed,

        # Quality
        "mean_likelihood":     d["mean_likelihood"].values,
    })
    all_frames.append(out)
    print(f"  {video[:55]:55s}  frames: {n_frames} (raw {n_raw})")


if not all_frames:
    raise SystemExit("Nothing extracted.")

frames_df = pd.concat(all_frames, ignore_index=True)

# --- Мета из имени видео ---
frames_df["mouse_parsed"] = frames_df["video"].str.split("_").str[0]
frames_df["group"]        = (frames_df["mouse_parsed"].str.lower()
                             .map(GROUP_BY_PREFIX).fillna("unknown"))

frames_df["condition"] = np.select(
    [frames_df["video"].str.contains("_black_", case=False, regex=False),
     frames_df["video"].str.contains("_stim",  case=False, regex=False)],
    ["baseline", "stim"], default="other")

frames_df["stim_dir"] = np.select(
    [frames_df["video"].str.endswith("_left"),
     frames_df["video"].str.endswith("_right")],
    ["left", "right"], default="")

out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
frames_df.to_csv(out_path, sep=";", decimal=",", index=False)

print(f"\nSaved: {out_path}")
print(f"  rows: {len(frames_df)}")
print(f"  by mouse:     {frames_df['mouse_parsed'].value_counts().to_dict()}")
print(f"  by group:     {frames_df['group'].value_counts().to_dict()}")
print(f"  by condition: {frames_df['condition'].value_counts().to_dict()}")