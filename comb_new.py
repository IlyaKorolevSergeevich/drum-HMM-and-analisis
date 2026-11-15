import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ---------- CONFIG ----------
FILE       = r"C:\Users\90108\OneDrive\Рабочий стол\c3h\5\UNKNOWN_c3h5_combined.csv"
OUTPUT_DIR = r"C:\Users\90108\OneDrive\Рабочий стол\v\agouty\frame_features10"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_NAME = "frame_features_c3h5_v3.csv"

FPS                  = 30
LIKELIHOOD_THRESHOLD = 0.70
SMOOTH_WINDOW        = 5      # final feature smoothing (frames)
POS_SMOOTH_WINDOW    = 7      # position smoothing before derivatives (frames)

REQUIRED_BODYPARTS = [
    "left_ear_base", "right_ear_base", "head_base", "tail_base",
]
GROUP_BY_PREFIX = {"c3h": "blind", "cba": "sighted"}
EPS = 1e-6


# ---------- HELPERS ----------
def wrap_angle(a):
    """Wrap angle(s) in degrees to [-180, 180)."""
    return (a + 180) % 360 - 180


def rolling_mean(series, w):
    """Centered rolling mean with edge handling (min_periods=1)."""
    return series.rolling(w, center=True, min_periods=1).mean()


def smooth_array(x, w):
    """Same as rolling_mean, but accepts a NumPy array."""
    return rolling_mean(pd.Series(x), w).values


# ---------- MAIN ----------
print("=" * 70)
print("FILE:", FILE)
print("=" * 70)

df = pd.read_csv(FILE, sep=";", decimal=",")

# Numeric coercion (European CSV may have ',' as decimal mark).
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
    d = df[df["video"] == video].copy()
    # Frame order matters for derivatives.
    if "frame" in d.columns:
        d = d.sort_values("frame").reset_index(drop=True)
    else:
        d = d.reset_index(drop=True)

    n_raw = len(d)
    if n_raw == 0:
        continue

    # --- Interpolation + per-frame quality masks ---------------------------
    # n_low_conf[i] = how many of the 4 required keypoints had low confidence
    #                 at frame i (before interpolation).
    # interp_mask[i] = True if at least one required keypoint was interpolated.
    interp_mask = np.zeros(n_raw, dtype=bool)
    n_low_conf  = np.zeros(n_raw, dtype=int)

    for bp in REQUIRED_BODYPARTS:
        lk_col = f"{bp}_likelihood"
        if lk_col not in d.columns:
            continue
        bad = (d[lk_col] < LIKELIHOOD_THRESHOLD).values
        interp_mask |= bad
        n_low_conf  += bad.astype(int)
        for ax in ("x", "y"):
            col = f"{bp}_{ax}"
            if col in d.columns:
                d.loc[bad, col] = np.nan
                d[col] = d[col].interpolate(limit_direction="both")

    # Frame is "valid" if at least 3 of 4 required points were reliable.
    frame_valid = (n_low_conf <= 1)

    # --- Mean likelihood as a quality feature ------------------------------
    lk_cols = [f"{bp}_likelihood" for bp in REQUIRED_BODYPARTS
               if f"{bp}_likelihood" in d.columns]
    mean_likelihood = (d[lk_cols].mean(axis=1).values
                       if lk_cols else np.full(n_raw, np.nan))

    # --- Smooth positions BEFORE computing derivatives ---------------------
    def pos(name):
        return smooth_array(d[name].values, POS_SMOOTH_WINDOW)

    Lx, Ly = pos("left_ear_base_x"),  pos("left_ear_base_y")
    Rx, Ry = pos("right_ear_base_x"), pos("right_ear_base_y")
    Hx, Hy = pos("head_base_x"),      pos("head_base_y")
    Tx, Ty = pos("tail_base_x"),      pos("tail_base_y")

    # --- Real time axis ----------------------------------------------------
    t = (d["frame"].values.astype(float) / FPS
         if "frame" in d.columns else np.arange(n_raw) / FPS)

    # --- Head geometry -----------------------------------------------------
    ear_mid_x, ear_mid_y = (Lx + Rx) / 2.0, (Ly + Ry) / 2.0
    ear_dx, ear_dy = Rx - Lx, Ry - Ly
    head_dx, head_dy = -ear_dy, ear_dx

    angle_raw  = np.degrees(np.arctan2(head_dy, head_dx))
    body_angle = np.degrees(np.arctan2(Hy - Ty, Hx - Tx))

    diff1 = np.abs(wrap_angle(angle_raw       - body_angle))
    diff2 = np.abs(wrap_angle(angle_raw + 180 - body_angle))
    head_angle = np.where(diff1 < diff2, angle_raw,
                          wrap_angle(angle_raw + 180))

    ha_rad          = np.radians(head_angle)
    head_angle_sin  = np.sin(ha_rad)
    head_angle_cos  = np.cos(ha_rad)

    hbd_rad            = np.radians(wrap_angle(head_angle - body_angle))
    head_body_diff_sin = np.sin(hbd_rad)
    head_body_diff_cos = np.cos(hbd_rad)

    # --- Size metrics ------------------------------------------------------
    head_area = 0.5 * np.abs((Lx - Hx) * (Ry - Hy) - (Rx - Hx) * (Ly - Hy))
    ear_line  = np.sqrt((Rx - Lx) ** 2 + (Ry - Ly) ** 2) + EPS
    head_len  = np.sqrt((ear_mid_x - Hx) ** 2 + (ear_mid_y - Hy) ** 2) + EPS

    head_area_norm = head_area / (ear_line * head_len)
    head_len_norm  = head_len / ear_line

    d_L = np.sqrt((Lx - Hx) ** 2 + (Ly - Hy) ** 2)
    d_R = np.sqrt((Rx - Hx) ** 2 + (Ry - Hy) ** 2)
    ear_asym_norm = (d_L - d_R) / (d_L + d_R + EPS)

    # --- Movement on the real time axis ------------------------------------
    hb_vx    = np.gradient(Hx, t)          # px / s
    hb_vy    = np.gradient(Hy, t)          # px / s
    hb_speed = np.sqrt(hb_vx ** 2 + hb_vy ** 2)

    d_angle = wrap_angle(np.diff(np.concatenate([[head_angle[0]], head_angle])))
    dt      = np.diff(np.concatenate([[t[0]], t]))
    dt      = np.where(dt > 0, dt, 1.0 / FPS)   # guard against zero / negative
    angular_speed = d_angle / dt                # deg / s

    out = pd.DataFrame({
        "video":               video,
        "frame":               (d["frame"].values if "frame" in d.columns
                                else np.arange(n_raw)),
        "time_s":              t,

        # Pose
        "head_area_norm":      head_area_norm,
        "head_len_norm":       head_len_norm,
        "ear_asym_norm":       ear_asym_norm,

        # Orientation (circular, sin/cos)
        "head_angle_sin":      head_angle_sin,
        "head_angle_cos":      head_angle_cos,
        "head_body_diff_sin":  head_body_diff_sin,
        "head_body_diff_cos":  head_body_diff_cos,

        # Movement
        "hb_speed":            hb_speed,
        "angular_speed":       angular_speed,

        # Quality
        "mean_likelihood":     mean_likelihood,
        "n_low_conf":          n_low_conf,
        "interp_mask":         interp_mask,
        "frame_valid":         frame_valid,
    })
    all_frames.append(out)
    print(f"  {video[:55]:55s}  frames: {n_raw}  "
          f"valid: {int(frame_valid.sum())}")


if not all_frames:
    raise SystemExit("Nothing extracted.")

frames_df = pd.concat(all_frames, ignore_index=True)

# --- Final smoothing, grouped by video (no cross-video leakage) ------------
SMOOTH_COLS = [
    "head_area_norm", "head_len_norm", "ear_asym_norm",
    "head_angle_sin", "head_angle_cos",
    "head_body_diff_sin", "head_body_diff_cos",
    "hb_speed", "angular_speed",
]

for c in SMOOTH_COLS:
    frames_df[c] = (
        frames_df.groupby("video", sort=False)[c]
                  .transform(lambda s: rolling_mean(s, SMOOTH_WINDOW))
    )

# --- Metadata from video name ---------------------------------------------
frames_df["mouse_parsed"] = frames_df["video"].str.split("_").str[0]
frames_df["group"] = (frames_df["mouse_parsed"].str.lower()
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
print(f"  rows:          {len(frames_df)}")
print(f"  valid frames:  {int(frames_df['frame_valid'].sum())} / {len(frames_df)}")
print(f"  by mouse:      {frames_df['mouse_parsed'].value_counts().to_dict()}")
print(f"  by group:      {frames_df['group'].value_counts().to_dict()}")
print(f"  by condition:  {frames_df['condition'].value_counts().to_dict()}")
