# Маппинг координат между камерами (top/bottom → door2)

## Метод

**Thin Plate Spline (TPS)** — `scipy.interpolate.RBFInterpolator` с ядром
`thin_plate_spline`, обученный отдельно для `top→door2` и `bottom→door2`.

Параметр `smoothing` подбирается grid search по GroupKFold CV (сессия целиком
попадает либо в train, либо в val — чтобы не было утечки между сессиями).

Протестированные альтернативы (все уступили TPS):
гомография, аффинное преобразование, полином deg 2–3, MLP, KNN,
локальный TPS, Per-pair Affine Ensemble, двухэтапная residual-коррекция.

## Финальные метрики (val, 14 сессий)

| Маппинг | MED | Медиана | p90 |
|---|---|---|---|
| top → door2 | **124.80 px** | 90.09 px | 258.39 px |
| bottom → door2 | **139.16 px** | 101.57 px | 280.93 px |

Подробные метрики — в файле `metrics.json`.

## Структура

```
solution/
├── train.py       — обучение, сохранение артефактов в artifacts/
├── predict.py     — загрузка артефактов, функция predict(x, y, source)
├── evaluate.py    — расчёт финальных метрик на val-сплите
├── diagnose.py    — диагностика: пространственная карта ошибок, per-session анализ
├── metrics.json   — финальные метрики val-сплита
├── requirements.txt
└── README.md
```

## Воспроизведение

```bash
# 1. Создать окружение и установить зависимости
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Обучение (датасет в ../coord_data относительно папки solution/)
python train.py --data_root ../coord_data --artifacts_dir artifacts

# 3. Финальная оценка на val → пересоздаёт metrics.json
python evaluate.py --data_root ../coord_data --artifacts_dir artifacts
```

## API

```python
from predict import Predictor

p = Predictor("artifacts")          # загрузить обученные модели

# Одна точка
x2, y2 = p.predict(743.96, 524.59, source="top")

# Пакет точек: (N, 2) → (N, 2)
import numpy as np
pts  = np.array([[743.96, 524.59], [920.42, 343.37]])
pred = p.predict_batch(pts, source="top")
```

```bash
# CLI
python predict.py --x 743.96 --y 524.59 --source top
python predict.py --x 900.0  --y 300.0  --source bottom
```

## Производительность и оптимизация inference

Замеры на CPU (N=3 835 обучающих точек):

| Режим | 1 точка | батч 1034 точки | размер модели |
|---|---|---|---|
| TPS (текущий) | 0.054 мс | 80 мс | 0.22 МБ |
| Grid 256×144 + bilinear | ~0.001 мс | 0.35 мс | 0.58 МБ |

Текущая реализация достаточна для большинства задач. Для high-throughput
inference (видео-поток 30 fps, сотни точек на кадр) рекомендуется
предвычислить предсказания TPS на регулярной сетке 256×144 пикселей
и заменить вызов модели билинейной интерполяцией по этой сетке —
ускорение ~230× при потере качества менее 3.3 px (val MED: 124.8 → ~125.2 px).
