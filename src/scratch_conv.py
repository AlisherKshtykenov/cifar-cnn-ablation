"""Свёртка, max pooling и формула размера выхода с нуля на numpy (по эталону «Convolution explained»).

Нужно для защиты теории: показать, что nn.Conv2d считает ровно то же самое.
"""
import numpy as np


def conv1d(x, k, mode="valid"):
    """Математическая 1D-свёртка: ядро разворачивается, затем скользит по входу.

    valid: только полные перекрытия, длина N-M+1
    same:  длина N (нули по краям)
    full:  длина N+M-1 (любое перекрытие)
    """
    x, k = np.asarray(x, float), np.asarray(k, float)[::-1]   # разворот ядра = отличие свёртки от корреляции
    n, m = len(x), len(k)
    pad = {"valid": 0, "same": (m - 1) // 2, "full": m - 1}[mode]
    xp = np.pad(x, pad)
    out_len = {"valid": n - m + 1, "same": n, "full": n + m - 1}[mode]
    return np.array([np.dot(xp[i:i + m], k) for i in range(out_len)])


def output_size(w, k, s=1, p=0):
    """Размер выхода свёртки/пулинга: floor((W - K + 2P) / S) + 1."""
    return (w - k + 2 * p) // s + 1


def conv2d(x, w, b=None, stride=1, padding=0):
    """Свёрточный слой CNN (как nn.Conv2d — это кросс-корреляция, ядро НЕ разворачивается).

    x: (C_in, H, W), w: (C_out, C_in, K, K) -> (C_out, H_out, W_out)
    Каждый выход = сумма по всем входным каналам поэлементного произведения окна и ядра + bias.
    """
    c_out, c_in, k, _ = w.shape
    x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    h_out = output_size(x.shape[1], k, stride)
    w_out = output_size(x.shape[2], k, stride)
    out = np.zeros((c_out, h_out, w_out))
    for o in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                patch = x[:, i * stride:i * stride + k, j * stride:j * stride + k]   # окно (C_in,K,K)
                out[o, i, j] = np.sum(patch * w[o]) + (b[o] if b is not None else 0.0)
    return out


def maxpool2d(x, k=2, stride=2):
    """Max pooling: максимум в каждом окне k x k, по каждому каналу отдельно."""
    c, h, w = x.shape
    h_out, w_out = output_size(h, k, stride), output_size(w, k, stride)
    out = np.zeros((c, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            out[:, i, j] = x[:, i * stride:i * stride + k, j * stride:j * stride + k].max(axis=(1, 2))
    return out
