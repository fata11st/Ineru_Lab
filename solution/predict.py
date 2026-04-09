"""
predict.py — загружает артефакты и предоставляет функцию predict(x, y, source).

Поддерживаемые типы артефактов: pae, knn, tps, mlp, poly, affine, homography.

Использование как скрипт:
    python predict.py --x 743.96 --y 524.59 --source top

Использование как модуль:
    from predict import Predictor
    p = Predictor("artifacts")
    x2, y2 = p.predict(743.96, 524.59, source="top")
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

IMG_W, IMG_H = 3200.0, 1800.0


class Predictor:
    def __init__(self, artifacts_dir="artifacts"):
        self._arts = {}
        for src in ("top", "bottom"):
            p = Path(artifacts_dir) / f"model_{src}.pkl"
            if not p.exists():
                raise FileNotFoundError(
                    f"Артефакт не найден: {p}\nЗапустите: python train.py")
            with open(p, "rb") as f:
                self._arts[src] = pickle.load(f)

    def predict(self, x: float, y: float, source: str):
        """
        Маппинг пиксельной координаты из top/bottom в door2.
        Returns: (x', y') на кадре door2.
        """
        if source not in ("top", "bottom"):
            raise ValueError(f"source: 'top' или 'bottom', получено {source!r}")
        pts = np.array([[x, y]], dtype=np.float64)
        res = self._predict_batch(pts, source)
        return float(res[0, 0]), float(res[0, 1])

    def predict_batch(self, pts: np.ndarray, source: str) -> np.ndarray:
        """pts: (N, 2) → (N, 2) в door2."""
        return self._predict_batch(np.asarray(pts, dtype=np.float64), source)

    def _predict_batch(self, pts, source):
        art = self._arts[source]
        t = art["type"]

        if t == "pae":
            return art["model"].predict(pts)

        elif t == "knn":
            return art["model"].predict(pts)

        elif t == "tps":
            return art["model"](pts)

        elif t == "tps_2stage":
            pred1 = art["model_primary"](pts)
            corr  = art["model_correction"](pts)
            return pred1 + corr

        elif t == "mlp":
            pts_n = pts / np.array([IMG_W, IMG_H])
            return art["model"].predict(pts_n) * np.array([IMG_W, IMG_H])

        elif t == "poly":
            return np.stack([art["pipe_x"].predict(pts),
                             art["pipe_y"].predict(pts)], axis=1)

        elif t == "affine":
            W = art["W"]
            pts_h = np.hstack([pts, np.ones((len(pts), 1))])
            return pts_h @ W

        elif t == "homography":
            H = art["H"]
            pts_h = np.hstack([pts, np.ones((len(pts), 1))])
            res = (H @ pts_h.T).T
            return res[:, :2] / res[:, 2:3]

        else:
            raise RuntimeError(f"Неизвестный тип модели: {t!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x",             type=float, required=True)
    parser.add_argument("--y",             type=float, required=True)
    parser.add_argument("--source",        required=True, choices=["top", "bottom"])
    parser.add_argument("--artifacts_dir", default="artifacts")
    args = parser.parse_args()

    p = Predictor(args.artifacts_dir)
    xp, yp = p.predict(args.x, args.y, args.source)
    print(f"Вход  ({args.source}): x={args.x:.2f}, y={args.y:.2f}")
    print(f"Выход  (door2): x={xp:.2f}, y={yp:.2f}")


if __name__ == "__main__":
    main()
