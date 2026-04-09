"""
diagnose.py — диагностика качества модели.

Отвечает на вопросы:
  1. В каких регионах src-изображения большие ошибки? (пространственная карта)
  2. Какие val-сессии дают плохой результат?
  3. Есть ли корреляция между ошибкой и расстоянием до ближайшей train-точки?
  4. Систематический ли bias ошибок (направление и величина)?
  5. Как выглядит распределение ошибок по перцентилям?

Использование:
    python diagnose.py --data_root coord_data --artifacts_dir artifacts
"""

import argparse
import json
from pathlib import Path

import numpy as np

from predict import Predictor


# ─────────────────────────── загрузка ─────────────────────────────────────

def load_split_data(data_root, sessions, source):
    """Загружает точки с метаданными о сессиях."""
    records = []
    for session in sessions:
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
                    records.append({
                        "session": session,
                        "src_x": sr[num][0], "src_y": sr[num][1],
                        "dst_x": d2[num][0], "dst_y": d2[num][1],
                    })
    return records


def sep(title="", width=60):
    if title:
        print(f"\n{'─'*4} {title} {'─'*(width - len(title) - 6)}")
    else:
        print("─" * width)


# ─────────────────────────── анализ ───────────────────────────────────────

def run_diagnostics(data_root, artifacts_dir):
    predictor = Predictor(artifacts_dir)

    with open(Path(data_root) / "split.json") as f:
        split = json.load(f)

    for source in ("top", "bottom"):
        print(f"\n{'═'*60}")
        print(f"  Диагностика: {source} → door2")
        print(f"{'═'*60}")

        train_recs = load_split_data(data_root, split["train"], source)
        val_recs   = load_split_data(data_root, split["val"],   source)

        if not val_recs:
            print("  Нет val-данных, пропускаем.")
            continue

        train_src = np.array([[r["src_x"], r["src_y"]] for r in train_recs])
        val_src   = np.array([[r["src_x"], r["src_y"]] for r in val_recs])
        val_dst   = np.array([[r["dst_x"], r["dst_y"]] for r in val_recs])
        val_pred  = predictor.predict_batch(val_src, source)
        val_err   = np.linalg.norm(val_pred - val_dst, axis=1)
        val_bias  = val_pred - val_dst   # вектор ошибки (направление)

        # ── 1. Общее распределение ошибок ──
        sep("1. Распределение ошибок")
        pcts = [10, 25, 50, 75, 90, 95, 99]
        vals = np.percentile(val_err, pcts)
        print(f"  {'Перцентиль':>12}  {'Ошибка (px)':>12}")
        for p, v in zip(pcts, vals):
            bar = "█" * int(v / 20)
            print(f"  {p:>11}%  {v:>10.1f}  {bar}")
        print(f"\n  MED    : {val_err.mean():.2f} px")
        print(f"  Медиана: {np.median(val_err):.2f} px")
        print(f"  Std    : {val_err.std():.2f} px")
        
        # Доля точек по порогам
        sep("2. Доля точек в пределах порога")
        thresholds = [50, 100, 150, 200, 300]
        for t in thresholds:
            pct = 100 * (val_err <= t).mean()
            bar = "█" * int(pct / 5)
            print(f"  ≤ {t:>3} px: {pct:>5.1f}%  {bar}")

        # ── 3. Систематический bias ──
        sep("3. Систематический bias (направление ошибок)")
        mean_bias_x = val_bias[:, 0].mean()
        mean_bias_y = val_bias[:, 1].mean()
        mean_bias_mag = np.linalg.norm([mean_bias_x, mean_bias_y])
        print(f"  Средний bias X: {mean_bias_x:+.2f} px")
        print(f"  Средний bias Y: {mean_bias_y:+.2f} px")
        print(f"  Магнитуда bias: {mean_bias_mag:.2f} px")
        if mean_bias_mag > 20:
            print("  ⚠ Значительный систематический сдвиг — возможно смещение камеры")
        else:
            print("  ✓ Bias мал — ошибки случайные, не систематические")

        # ── 4. Per-session анализ ──
        sep("4. Ошибка по val-сессиям")
        sessions_val = [r["session"] for r in val_recs]
        unique_sessions = sorted(set(sessions_val))
        session_meds = {}
        for sess in unique_sessions:
            mask = np.array([s == sess for s in sessions_val])
            sess_err = val_err[mask]
            session_meds[sess] = sess_err.mean()

        sorted_sessions = sorted(session_meds.items(), key=lambda x: x[1])
        overall_med = val_err.mean()
        print(f"  {'Сессия':<45} {'MED':>8}  {'vs avg':>8}")
        for sess, m in sorted_sessions:
            name = Path(sess).name[-25:]
            diff = m - overall_med
            flag = "  ⚠" if diff > 50 else ("  ✓" if diff < -30 else "")
            print(f"  {name:<45} {m:>7.1f}  {diff:>+8.1f}{flag}")

        worst_sessions = [s for s, m in sorted_sessions if m > overall_med + 50]
        if worst_sessions:
            print(f"\n  ⚠ Проблемных сессий: {len(worst_sessions)} (MED > avg+50px)")
        
        # ── 5. Ошибка vs покрытие train-данными ──
        sep("5. Ошибка vs расстояние до ближайшей train-точки")
        # Для каждой val-точки — расстояние до ближайшей train-точки
        from sklearn.neighbors import BallTree
        bt = BallTree(train_src)
        nn_dist, _ = bt.query(val_src, k=1)
        nn_dist = nn_dist[:, 0]

        # Разбить на квантильные корзины по расстоянию
        bins = np.quantile(nn_dist, [0, 0.25, 0.5, 0.75, 1.0])
        print(f"  {'Расст. до train':>18}  {'Ср. ошибка':>12}  {'Кол-во':>8}")
        for i in range(len(bins) - 1):
            mask = (nn_dist >= bins[i]) & (nn_dist < bins[i+1])
            if mask.sum() == 0: continue
            avg_err = val_err[mask].mean()
            label = f"{bins[i]:.0f}–{bins[i+1]:.0f} px"
            bar = "█" * int(avg_err / 20)
            print(f"  {label:>18}  {avg_err:>10.1f}  {mask.sum():>8}  {bar}")

        # Корреляция расстояние-ошибка
        corr = np.corrcoef(nn_dist, val_err)[0, 1]
        print(f"\n  Корреляция (расст., ошибка): {corr:.3f}")
        if corr > 0.3:
            print("  ⚠ Сильная корреляция — улучшение покрытия train-данных даст эффект")
        else:
            print("  ✓ Слабая корреляция — дело не в покрытии")

        # ── 6. Пространственная карта ошибок (ASCII) ──
        sep("6. Пространственная карта ошибок в src-пространстве")
        # Сетка 8×5 ячеек
        nx, ny = 8, 5
        x_edges = np.linspace(val_src[:, 0].min(), val_src[:, 0].max(), nx + 1)
        y_edges = np.linspace(val_src[:, 1].min(), val_src[:, 1].max(), ny + 1)
        grid = np.full((ny, nx), np.nan)
        count = np.zeros((ny, nx), dtype=int)

        for xi in range(nx):
            for yi in range(ny):
                mask = (
                    (val_src[:, 0] >= x_edges[xi]) & (val_src[:, 0] < x_edges[xi+1]) &
                    (val_src[:, 1] >= y_edges[yi]) & (val_src[:, 1] < y_edges[yi+1])
                )
                if mask.sum() > 0:
                    grid[yi, xi] = val_err[mask].mean()
                    count[yi, xi] = mask.sum()

        # ASCII-визуализация
        levels = [0, 75, 125, 175, 250, 9999]
        chars  = [" ", "░", "▒", "▓", "█"]
        legend = "  Ошибка:  " + "  ".join(
            f"{chars[i]}={levels[i]}–{levels[i+1]}px" for i in range(len(chars))
        )
        print(f"  (строки = Y, столбцы = X в пространстве src-камеры)")
        print()
        for yi in range(ny):
            row = "  │"
            for xi in range(nx):
                if np.isnan(grid[yi, xi]):
                    row += "  "
                else:
                    v = grid[yi, xi]
                    for li, (lo, hi) in enumerate(zip(levels[:-1], levels[1:])):
                        if lo <= v < hi:
                            row += chars[li] * 2
                            break
            row += "│"
            print(row)
        print(f"  └{'─'*nx*2}┘")
        print(f"\n{legend}")

        # ── 7. Вывод: где есть потенциал ──
        sep("7. Итоговые выводы")
        print(f"  Val MED:       {val_err.mean():.1f} px")
        print(f"  Val медиана:   {np.median(val_err):.1f} px")
        gap = val_err.mean() - np.median(val_err)
        print(f"  Mean−медиана:  {gap:.1f} px  ({'выбросы тянут MED вверх' if gap > 20 else 'распределение симметричное'})")
        pct_bad = 100 * (val_err > 200).mean()
        print(f"  Доля > 200px:  {pct_bad:.1f}%")
        
        if pct_bad > 10:
            print("\n  → Главный резерв: снизить хвост (точки > 200px)")
            print("    Варианты: outlier detection на val, ensembling, per-session адаптация")
        if len(worst_sessions) > 2:
            print(f"\n  → {len(worst_sessions)} плохих сессий — возможно нужна per-session модель")
        if corr > 0.3:
            print(f"\n  → Добавить данные в слабо покрытые регионы")
        if mean_bias_mag < 15:
            print("\n  → Bias мал, общая модель правильно откалибрована")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     default="coord_data")
    parser.add_argument("--artifacts_dir", default="artifacts")
    args = parser.parse_args()
    run_diagnostics(args.data_root, args.artifacts_dir)
