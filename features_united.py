"""
Merge several frame-feature CSVs (e.g. frame_features_c3h2_v2.csv,
frame_features_cba3_v2.csv, ...) into one combined CSV for
hmm_train_frames_all.py.

The script can be run from anywhere: it looks for input files in the
folder specified in CONFIG below.

Group (blind / sighted) is derived from the video-name prefix:
    c3h* -> blind
    cba* -> sighted
Columns mouse_parsed and group are recomputed when requested so that
their values are always consistent with the naming convention.
"""

import os
import glob
import pandas as pd

# ============================================================
# CONFIG  —  EDIT PATHS HERE
# ============================================================

# Folder that contains the source CSVs
# (e.g. frame_features_c3h2_v2.csv, ...)
INPUT_DIR = r"C:\Users\90108\OneDrive\Рабочий стол\v\agouty\frame_features10"

# Glob pattern used to find the input files inside INPUT_DIR
PATTERN = "frame_features_*.csv"

# Output file (full path, including file name)
OUTPUT_FILE = r"C:\Users\90108\OneDrive\Рабочий стол\v\agouty\frame_features\frame_features10.csv"

# Whether to recompute mouse_parsed and group from the video name
RECOMPUTE_GROUP = True

# Video-name prefix -> group
GROUP_BY_PREFIX = {
    "c3h": "blind",
    "cba": "sighted",
}

# ============================================================
# HELPERS
# ============================================================

def get_group_from_video(video_name: str) -> str:
    """Return 'blind', 'sighted' or 'unknown' based on the video prefix."""
    if not isinstance(video_name, str):
        return "unknown"
    name = video_name.strip().lower()
    for prefix, group in GROUP_BY_PREFIX.items():
        if name.startswith(prefix.lower()):
            return group
    return "unknown"


def get_mouse_from_video(video_name: str) -> str:
    """Extract the mouse id from the video name (part before the first '_')."""
    if not isinstance(video_name, str):
        return "unknown"
    return video_name.split("_")[0]


# ============================================================
# MAIN LOGIC
# ============================================================

def merge_csvs(input_dir: str,
               pattern: str,
               output_file: str,
               recompute_group: bool = True) -> None:
    search_path = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(search_path))

    if not files:
        raise FileNotFoundError(
            f"No files matched the pattern: {search_path}\n"
            f"Check INPUT_DIR and PATTERN in CONFIG."
        )

    print(f"Search folder: {input_dir}")
    print(f"Pattern:       {pattern}")
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {os.path.basename(f)}")

    dfs = []
    for f in files:
        print(f"Reading: {os.path.basename(f)} ...", end=" ")
        try:
            df = pd.read_csv(f, sep=";", decimal=",")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        print(f"{len(df)} rows")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("Could not read any of the input files.")

    merged = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"\nTotal rows after concatenation: {len(merged)}")

    required = ["video", "frame", "time_s"]
    for col in required:
        if col not in merged.columns:
            raise KeyError(f"Column '{col}' is missing in the merged data")

    if recompute_group:
        print("Recomputing mouse_parsed and group from video names...")
        merged["mouse_parsed"] = merged["video"].apply(get_mouse_from_video)
        merged["group"] = merged["video"].apply(get_group_from_video)

    preferred_order = [
        "video", "frame", "time_s",
        "head_area_norm", "head_len_norm", "ear_asym_norm",
        "head_angle_sin", "head_angle_cos",
        "head_body_diff_sin", "head_body_diff_cos",
        "hb_speed", "angular_speed", "mean_likelihood",
        "mouse_parsed", "group", "condition", "stim_dir"
    ]
    ordered_cols = [c for c in preferred_order if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in ordered_cols]
    merged = merged[ordered_cols + other_cols]

    # Create the output folder if it does not exist yet
    out_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(output_file, sep=";", decimal=",", index=False)
    print(f"\nResult saved to: {os.path.abspath(output_file)}")
    print(f"Columns: {list(merged.columns)}")
    print(f"Rows: {len(merged)}")
    print("Groups:")
    print(merged["group"].value_counts().to_string())
    print("Mice:")
    print(merged["mouse_parsed"].value_counts().to_string())


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    merge_csvs(
        input_dir=INPUT_DIR,
        pattern=PATTERN,
        output_file=OUTPUT_FILE,
        recompute_group=RECOMPUTE_GROUP,
    )
