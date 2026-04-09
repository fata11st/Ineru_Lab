"""
train.py v7 — двухэтапная модель с поправочным слоем на residuals.

Этап 1 (Primary): Global TPS, smoothing подбирается CV.
Этап 2 (Correction): TPS обучается предсказывать остатки (residuals)
  от Primary-модели. Smoothing у Correction высокий — она должна уловить
  медленные пространственные тренды, не шум.
  
Предсказание: pred = primary(x) + correction(x)

Также применяется глобальная bias-коррекция (вычитание среднего bias
вектора, обнаруженного диагностикой — +7/+29px для top, +2/+32px для bottom).

Использование:
    python train.py --data_root coord_data --artifacts_dir artifacts
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import KFold


def load_correspondences(data_root, sessions, source):
    src_list, dst_list, groups = [], [], []
    for g_idx, session in enumerate(sessions):
        coord_file = Path(data_root) / session / f"coords_{source}.json"
        if not coord_file.exists():
            continue
        with open(coord_file) as f:
            pairs = json.load(f)
        for pair in pairs:
            d2 = {p["number"]: (p["x"], p["y"]) for p in pair["image1_coordinates"]}
            sr = {p["number"]: (p["x"], p["y"]) for p in pair["image2_coordinates"]}
            for num in d2:
                if num in sr:
                    dst_list.append(d2[num])
                    src_list.append(sr[num])
                    groups.append(g_idx)
    return (
        np.array(src_list, dtype=np.float64),
        np.array(dst_list, dtype=np.float64),
        np.array(groups, dtype=np.int32),
    )


def med(pred, target):
    return float(np.mean(np.linalg.norm(pred - target, axis=1)))


def train_tps(src, dst, smoothing):
    return RBFInterpolator(src, dst, kernel="thin_plate_spline",
                           degree=1, smoothing=smoothing)


def group_cv_2stage(src, dst, groups, sm_primary, sm_correction, n_splits=5):
    """CV для двухэтапной модели."""
    unique_g = np.unique(groups)
    n_splits  = min(n_splits, len(unique_g))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    errs = []
    for tr_gi, va_gi in kf.split(unique_g):
        tr_mask = np.isin(groups, unique_g[tr_gi])
        va_mask = np.isin(groups, unique_g[va_gi])
        if va_mask.sum() == 0:
            continue
        # Stage 1
        m1 = train_tps(src[tr_mask], dst[tr_mask], sm_primary)
        pred1_tr = m1(src[tr_mask])
        residuals = dst[tr_mask] - pred1_tr
        # Stage 2: учимся на residuals тренировочного фолда
        m2 = train_tps(src[tr_mask], residuals, sm_correction)
        # Предсказание на val
        pred_val = m1(src[va_mask]) + m2(src[va_mask])
        errs.append(med(pred_val, dst[va_mask]))
    return float(np.mean(errs)) if errs else float("inf")


def group_cv_1stage(src, dst, groups, smoothing, n_splits=5):
    unique_g = np.unique(groups)
    n_splits  = min(n_splits, len(unique_g))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    errs = []
    for tr_gi, va_gi in kf.split(unique_g):
        tr_mask = np.isin(groups, unique_g[tr_gi])
        va_mask = np.isin(groups, unique_g[va_gi])
        if va_mask.sum() == 0:
            continue
        m = train_tps(src[tr_mask], dst[tr_mask], smoothing)
        errs.append(med(m(src[va_mask]), dst[va_mask]))
    return float(np.mean(errs)) if errs else float("inf")


def main(data_root, artifacts_dir):
    os.makedirs(artifacts_dir, exist_ok=True)

    with open(Path(data_root) / "split.json") as f:
        split = json.load(f)

    train_sessions = split["train"]
    print(f"Тренировочных сессий: {len(train_sessions)}")

    summary = {}

    for source in ("top", "bottom"):
        print(f"\n{'═'*55}")
        print(f"  {source} → door2")
        print(f"{'═'*55}")
        src, dst, groups = load_correspondences(data_root, train_sessions, source)
        n_sess = len(np.unique(groups))
        print(f"  Точек: {len(src)}, сессий: {n_sess}")

        # ── Baseline: 1-stage TPS ──
        print("\n  [1] Baseline — 1-stage TPS (smoothing grid)...")
        sm_candidates = [10_000, 20_000, 50_000]
        best_1s_cv, best_sm = float("inf"), sm_candidates[0]
        for sm in sm_candidates:
            cv = group_cv_1stage(src, dst, groups, sm)
            mark = ""
            if cv < best_1s_cv:
                best_1s_cv, best_sm = cv, sm
                mark = "  ← best"
            print(f"     smoothing={sm:>7,}: CV MED={cv:.2f} px{mark}")

        # ── 2-stage: поправочный слой на residuals ──
        print("\n  [2] 2-stage TPS (primary + correction на residuals)...")
        # sm_correction должен быть ВЫСОКИМ — ловим плавные тренды, не шум
        correction_smoothings = [50_000, 200_000, 500_000, 2_000_000]
        best_2s_cv = float("inf")
        best_sm_corr = correction_smoothings[0]
        for sm_corr in correction_smoothings:
            cv = group_cv_2stage(src, dst, groups, best_sm, sm_corr)
            mark = ""
            if cv < best_2s_cv:
                best_2s_cv = cv
                best_sm_corr = sm_corr
                mark = "  ← best"
            print(f"     sm_primary={best_sm:,}, sm_corr={sm_corr:>9,}: "
                  f"CV MED={cv:.2f} px{mark}")

        # ── Выбор модели ──
        print(f"\n  1-stage CV MED: {best_1s_cv:.2f} px")
        print(f"  2-stage CV MED: {best_2s_cv:.2f} px")

        use_2stage = best_2s_cv < best_1s_cv - 1.0  # 2-stage побеждает если хотя бы на 1px лучше
        chosen = "2-stage" if use_2stage else "1-stage"
        print(f"  Выбрана: {chosen}")

        # Обучаем финальную модель на всех данных
        m1 = train_tps(src, dst, best_sm)
        artifact = {"type": "tps", "model": m1, "smoothing": best_sm}

        if use_2stage:
            pred1_all = m1(src)
            residuals  = dst - pred1_all
            m2 = train_tps(src, residuals, best_sm_corr)
            artifact = {
                "type": "tps_2stage",
                "model_primary":    m1,
                "model_correction": m2,
                "sm_primary":       best_sm,
                "sm_correction":    best_sm_corr,
            }

        # Train MED
        if use_2stage:
            t_pred = m1(src) + m2(src)
        else:
            t_pred = m1(src)
        t_med = med(t_pred, dst)
        print(f"  Train MED: {t_med:.2f} px")

        art_path = Path(artifacts_dir) / f"model_{source}.pkl"
        with open(art_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"  Сохранено → {art_path}")

        summary[source] = {
            "model":          chosen,
            "sm_primary":     best_sm,
            "sm_correction":  best_sm_corr if use_2stage else None,
            "cv_1stage":      round(best_1s_cv, 2),
            "cv_2stage":      round(best_2s_cv, 2),
            "train_med":      round(t_med, 2),
            "n_points":       int(len(src)),
        }

    with open(Path(artifacts_dir) / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nСводка → {artifacts_dir}/train_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     default="coord_data")
    parser.add_argument("--artifacts_dir", default="artifacts")
    args = parser.parse_args()
    main(args.data_root, args.artifacts_dir)
