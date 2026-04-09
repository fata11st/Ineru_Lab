"""
evaluate.py — считает MED на валидационном сплите и сохраняет metrics.json.

Использование:
    python evaluate.py --data_root coord_data --artifacts_dir artifacts
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from predict import Predictor


def load_val_correspondences(data_root: str, sessions: list[str], source: str):
    """Загружает все точки соответствия из val-сессий."""
    src_pts, dst_pts = [], []
    session_errors = []

    for session in sessions:
        coord_file = Path(data_root) / session / f"coords_{source}.json"
        if not coord_file.exists():
            print(f"  [warn] не найден {coord_file}")
            continue

        with open(coord_file) as f:
            pairs = json.load(f)

        session_src, session_dst = [], []
        for pair in pairs:
            door2_by_num = {p["number"]: (p["x"], p["y"])
                            for p in pair["image1_coordinates"]}
            src_by_num   = {p["number"]: (p["x"], p["y"])
                            for p in pair["image2_coordinates"]}

            for num in door2_by_num:
                if num in src_by_num:
                    session_dst.append(door2_by_num[num])
                    session_src.append(src_by_num[num])

        src_pts.extend(session_src)
        dst_pts.extend(session_dst)
        session_errors.append({
            "session": session,
            "n_points": len(session_src),
        })

    return (
        np.array(src_pts, dtype=np.float64),
        np.array(dst_pts, dtype=np.float64),
        session_errors,
    )


def evaluate_source(predictor: Predictor, data_root: str,
                    val_sessions: list[str], source: str) -> dict:
    print(f"\n── Оценка {source} → door2 ──")
    src, dst, session_info = load_val_correspondences(data_root, val_sessions, source)
    print(f"  Точек в val: {len(src)}")

    pred = predictor.predict_batch(src, source)
    errors = np.linalg.norm(pred - dst, axis=1)

    result = {
        "MED_px":        round(float(errors.mean()), 3),
        "median_px":     round(float(np.median(errors)), 3),
        "std_px":        round(float(errors.std()), 3),
        "p90_px":        round(float(np.percentile(errors, 90)), 3),
        "p95_px":        round(float(np.percentile(errors, 95)), 3),
        "n_points":      int(len(src)),
        "n_sessions":    len(val_sessions),
    }

    print(f"  MED     : {result['MED_px']:.2f} px")
    print(f"  median  : {result['median_px']:.2f} px")
    print(f"  std     : {result['std_px']:.2f} px")
    print(f"  p90     : {result['p90_px']:.2f} px")
    print(f"  p95     : {result['p95_px']:.2f} px")

    return result


def main(data_root: str, artifacts_dir: str, output_path: str):
    split_path = Path(data_root) / "split.json"
    with open(split_path) as f:
        split = json.load(f)

    val_sessions = split["val"]
    print(f"Val сессий: {len(val_sessions)}")

    predictor = Predictor(artifacts_dir)

    metrics = {}
    for source in ("top", "bottom"):
        metrics[f"{source}_to_door2"] = evaluate_source(
            predictor, data_root, val_sessions, source
        )

    # Итоговый вывод
    print("\n══════════════ ФИНАЛЬНЫЕ МЕТРИКИ (val) ══════════════")
    print(f"  top   → door2  MED: {metrics['top_to_door2']['MED_px']:.2f} px")
    print(f"  bottom→ door2  MED: {metrics['bottom_to_door2']['MED_px']:.2f} px")
    print("═════════════════════════════════════════════════════")

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nМетрики сохранены → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     default="coord_data")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--output",        default="metrics.json",
                        help="путь к выходному файлу с метриками")
    args = parser.parse_args()
    main(args.data_root, args.artifacts_dir, args.output)
