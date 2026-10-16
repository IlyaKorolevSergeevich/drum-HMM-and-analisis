# merge_frame_features.py
# -*- coding: utf-8 -*-
"""
Объединение нескольких CSV-файлов с покадровыми признаками
(например, frame_features_c3h2_v2.csv, frame_features_cba3_v2.csv и т.п.)
в один общий CSV для скрипта hmm_train_frames_all.py.

Скрипт можно запускать откуда угодно — он сам ищет файлы в папке,
указанной в CONFIG ниже.

Группа (blind / sighted) определяется по префиксу имени видео:
    c3h* -> blind
    cba* -> sighted
Столбцы mouse_parsed и group при необходимости пересчитываются,
чтобы гарантировать корректность.
"""

import os
import glob
import pandas as pd

# ============================================================
# CONFIG  —  ЗДЕСЬ МЕНЯЙ ПУТИ
# ============================================================

# Папка, в которой лежат исходные CSV (например, frame_features_c3h2_v2.csv, ...)
INPUT_DIR = r"C:\Users\90108\OneDrive\Рабочий стол\v\agouty\frame_features10"

# Маска для поиска файлов в INPUT_DIR
PATTERN = "frame_features_*.csv"

# Куда сохранить объединённый файл (полный путь)
OUTPUT_FILE = r"C:\Users\90108\OneDrive\Рабочий стол\v\agouty\frame_features\frame_features10.csv"

# Пересчитывать ли mouse_parsed и group по имени видео
RECOMPUTE_GROUP = True

# Префиксы видео -> группа
GROUP_BY_PREFIX = {
    "c3h": "blind",
    "cba": "sighted",
}

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_group_from_video(video_name: str) -> str:
    """Возвращает 'blind', 'sighted' или 'unknown' по префиксу видео."""
    if not isinstance(video_name, str):
        return "unknown"
    name = video_name.strip().lower()
    for prefix, group in GROUP_BY_PREFIX.items():
        if name.startswith(prefix.lower()):
            return group
    return "unknown"


def get_mouse_from_video(video_name: str) -> str:
    """Извлекает имя мыши из имени видео (часть до первого '_')."""
    if not isinstance(video_name, str):
        return "unknown"
    return video_name.split("_")[0]


# ============================================================
# ОСНОВНАЯ ЛОГИКА
# ============================================================

def merge_csvs(input_dir: str,
               pattern: str,
               output_file: str,
               recompute_group: bool = True) -> None:
    search_path = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(search_path))

    if not files:
        raise FileNotFoundError(
            f"Не найдено ни одного файла по шаблону: {search_path}\n"
            f"Проверь INPUT_DIR и PATTERN в CONFIG."
        )

    print(f"Папка поиска: {input_dir}")
    print(f"Маска:        {pattern}")
    print(f"Найдено {len(files)} файлов:")
    for f in files:
        print(f"  {os.path.basename(f)}")

    dfs = []
    for f in files:
        print(f"Чтение: {os.path.basename(f)} ...", end=" ")
        try:
            df = pd.read_csv(f, sep=";", decimal=",")
        except Exception as e:
            print(f"ОШИБКА: {e}")
            continue
        print(f"{len(df)} строк")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("Не удалось прочитать ни одного файла.")

    merged = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"\nВсего строк после объединения: {len(merged)}")

    required = ["video", "frame", "time_s"]
    for col in required:
        if col not in merged.columns:
            raise KeyError(f"В объединённых данных отсутствует столбец '{col}'")

    if recompute_group:
        print("Пересчёт mouse_parsed и group по имени видео...")
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

    # Создаём папку для выходного файла, если её нет
    out_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(output_file, sep=";", decimal=",", index=False)
    print(f"\nРезультат сохранён в: {os.path.abspath(output_file)}")
    print(f"Столбцы: {list(merged.columns)}")
    print(f"Строк: {len(merged)}")
    print("Группы:")
    print(merged["group"].value_counts().to_string())
    print("Мыши:")
    print(merged["mouse_parsed"].value_counts().to_string())


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    merge_csvs(
        input_dir=INPUT_DIR,
        pattern=PATTERN,
        output_file=OUTPUT_FILE,
        recompute_group=RECOMPUTE_GROUP,
    )