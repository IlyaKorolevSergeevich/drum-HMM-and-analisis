# -*- coding: utf-8 -*-
"""
Build combined.csv from individual DLC *_filtered.csv files.

Robust version:
- auto-detects separator ("," or ";")
- auto-detects header structure (1, 2, or 3 levels)
- handles both MultiIndex and flat column names
- works with DLC 2.2, 2.3, 2.4+
"""

import os
import glob
import re
import pandas as pd


# =====================================================
# ROOT FOLDER
# =====================================================

ROOT_FOLDER = r"C:\Users\90108\OneDrive\Рабочий стол\cba"



# =====================================================
# HELPER: detect separator
# =====================================================

def detect_separator(file_path):
    """Detect CSV separator by reading the first line."""
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline()

    # Count commas vs semicolons
    n_comma = first_line.count(",")
    n_semi = first_line.count(";")

    if n_semi > n_comma:
        return ";"
    return ","


# =====================================================
# HELPER: detect header levels
# =====================================================

def detect_header_levels(file_path, sep):
    """
    Try reading with 1, 2, and 3 header levels.
    Return the one that produces the most sensible structure.
    """
    for levels in [3, 2, 1]:
        try:
            if levels == 1:
                df = pd.read_csv(file_path, sep=sep, header=0, nrows=5)
            else:
                df = pd.read_csv(
                    file_path,
                    sep=sep,
                    header=list(range(levels)),
                    nrows=5
                )

            # If we got at least one column named with "x", "y", "likelihood" — good
            col_strings = [str(c).lower() for c in df.columns]
            has_coord = any(
                ("x" in c and "y" not in c) or "likelihood" in c or "_x" in c or "_y" in c
                for c in col_strings
            )
            if has_coord:
                return levels
        except Exception:
            continue

    return 1  # fallback


# =====================================================
# HELPER: flatten columns
# =====================================================

def flatten_columns(df):
    """
    Convert any column structure (MultiIndex or flat) to flat strings
    like 'nose_x', 'nose_y', 'nose_likelihood'.
    """
    new_cols = []

    for col in df.columns:

        # Already a string -> keep as is
        if isinstance(col, str):
            new_cols.append(col)
            continue

        # Tuple (MultiIndex) -> join meaningfully
        parts = [str(p) for p in col if str(p).lower() not in
                 ("nan", "none", "")]

        # Find the bodypart and coord
        bodypart = None
        coord = None

        for p in parts:
            pl = p.lower()
            if pl in ("x", "y", "likelihood"):
                coord = pl
            elif pl == "coords":
                coord = "coords"
            elif pl == "bodyparts":
                continue
            elif pl.startswith("dlc") or pl.startswith("resnet") or pl.startswith("snapshot"):
                continue
            else:
                if bodypart is None:
                    bodypart = p

        if bodypart is None:
            new_cols.append("_".join(parts) if parts else "unnamed")
        elif coord is None:
            new_cols.append(bodypart)
        else:
            new_cols.append(f"{bodypart}_{coord}")

    df.columns = new_cols
    return df


# =====================================================
# FIND SUBFOLDERS
# =====================================================

subfolders = [
    f.path
    for f in os.scandir(ROOT_FOLDER)
    if f.is_dir()
]

print(f"Found {len(subfolders)} folders")
print()


# =====================================================
# PROCESS EACH FOLDER
# =====================================================

for folder in subfolders:

    folder_name = os.path.basename(folder)

    print("=" * 60)
    print(folder_name)

    files = sorted(glob.glob(os.path.join(folder, "*filtered.csv")))

    if len(files) == 0:
        print("  No filtered files -> skipped")
        print()
        continue

    all_data = []
    skipped = 0

    for file in files:

        print(f"  Reading: {os.path.basename(file)}")

        # ---- detect separator ----
        sep = detect_separator(file)

        # ---- detect header levels ----
        levels = detect_header_levels(file, sep)

        # ---- read with detected settings ----
        try:
            if levels == 1:
                df = pd.read_csv(file, sep=sep, header=0)
            else:
                df = pd.read_csv(
                    file,
                    sep=sep,
                    header=list(range(levels))
                )
        except Exception as e:
            print(f"    ERROR reading: {e}")
            skipped += 1
            continue

        print(f"    sep='{sep}', header_levels={levels}, shape={df.shape}")

        # ---- flatten columns ----
        try:
            df = flatten_columns(df)
        except Exception as e:
            print(f"    ERROR flattening columns: {e}")
            skipped += 1
            continue

        # ---- drop 'coords' row if present ----
        if "bodyparts_coords" in df.columns:
            df = df[df["bodyparts_coords"] != "coords"]

        # ---- drop rows that are just headers repeated ----
        # Sometimes DLC writes the header twice (multi-scorer)
        if "bodyparts_coords" in df.columns:
            df = df[df["bodyparts_coords"].astype(str).str.lower() != "coords"]

        df = df.reset_index(drop=True)

        # ---- sanity check: do we have x/y/likelihood columns? ----
        has_x = any(c.endswith("_x") for c in df.columns)
        has_y = any(c.endswith("_y") for c in df.columns)
        has_lik = any(c.endswith("_likelihood") for c in df.columns)

        if not (has_x and has_y and has_lik):
            print(f"    WARNING: missing expected columns (x={has_x}, y={has_y}, lik={has_lik})")
            print(f"    Columns: {df.columns.tolist()[:10]}")

        # ---- add video and frame columns ----
        video_name = os.path.splitext(os.path.basename(file))[0]

        if "video" in df.columns:
            df = df.drop(columns=["video"])
        if "frame" in df.columns:
            df = df.drop(columns=["frame"])

        df.insert(0, "video", video_name)
        df.insert(1, "frame", range(len(df)))

        all_data.append(df)
        print(f"    OK ({len(df)} rows)")

    if len(all_data) == 0:
        print("  No valid files -> skipped")
        print()
        continue

    # =================================================
    # CONCAT
    # =================================================

    try:
        result = pd.concat(all_data, ignore_index=True, sort=False)
    except Exception as e:
        print(f"  ERROR concatenating: {e}")
        print()
        continue

    # =================================================
    # EXTRACT DATE
    # =================================================

    first_video = str(result["video"].iloc[0])

    # Try several date patterns
    date = "UNKNOWN"

    m = re.search(r'_(\d{8})_', first_video)
    if m:
        date = m.group(1)
    else:
        m = re.search(r'(\d{8})', first_video)
        if m:
            date = m.group(1)
        else:
            m = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', first_video)
            if m:
                date = m.group(1).replace("-", "").replace("_", "")

    # =================================================
    # SAVE
    # =================================================

    outfile = os.path.join(
        folder,
        f"{date}_{folder_name}_combined.csv"
    )

    try:
        result.to_csv(
            outfile,
            sep=";",
            decimal=",",
            index=False,
            encoding="utf-8-sig"
        )
        print()
        print(f"  Saved: {os.path.basename(outfile)}")
        print(f"  Total rows: {len(result)}")
        print(f"  Columns: {len(result.columns)}")
    except PermissionError:
        print()
        print(f"  ERROR: cannot save — file is open in another program")
        print(f"  Path: {outfile}")
    except Exception as e:
        print()
        print(f"  ERROR saving: {e}")

    print()


# =====================================================
# DONE
# =====================================================

print("=" * 60)
print("DONE")