"""Загрузка CIFAR-10/100 из parquet (зеркало Hugging Face) и разбиение train/val/test."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def _decode(df, label_col):
    """PNG-байты из parquet -> массив uint8 (N, 32, 32, 3) и метки int64."""
    x = np.stack([np.array(Image.open(io.BytesIO(r["bytes"])).convert("RGB")) for r in df["img"]])
    y = df[label_col].to_numpy(dtype=np.int64)
    return x, y


def load_cifar(name="cifar10"):
    """Возвращает (x_train, y_train, x_test, y_test) в uint8.

    Сырые parquet не меняются; декодированные массивы кэшируются в data/processed/.
    """
    cache = DATA_PROCESSED / f"{name}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    label_col = "label" if name == "cifar10" else "fine_label"
    folder = DATA_RAW / f"hf_{name}"
    x_train, y_train = _decode(pd.read_parquet(folder / "train.parquet"), label_col)
    x_test, y_test = _decode(pd.read_parquet(folder / "test.parquet"), label_col)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return x_train, y_train, x_test, y_test


def split_train_val(x, y, val_size=5000, seed=42):
    """Стратифицированно отделяет валидацию от train: тест не используется для выбора модели."""
    return train_test_split(x, y, test_size=val_size, stratify=y, random_state=seed)


def channel_stats(x_uint8):
    """Среднее и std по каналам RGB в шкале [0, 1]. Считаются только на train (нет утечки).

    Считаем в float64: сумма 46 млн чисел во float32 теряет точность (получалось среднее 0.32768).
    """
    x = x_uint8.reshape(-1, 3).astype(np.float64) / 255.0
    return x.mean(axis=0).astype(np.float32), x.std(axis=0).astype(np.float32)
