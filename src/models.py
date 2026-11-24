"""Модели: точные копии эталонных архитектур 10 (Keras -> PyTorch) и улучшенные варианты.

Отличие от Keras: последний слой отдаёт logits без softmax, потому что
nn.CrossEntropyLoss сам делает log_softmax (так численно устойчивее).
Keras 'sparse_categorical_crossentropy' + softmax = то же самое.
"""
import torch
from torch import nn


# ---------- 1. MLP: картинка -> вектор 3072 чисел ----------
class LectureMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                       # (3, 32, 32) -> 3072, пространственная структура теряется
            nn.Linear(3072, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------- 2. Простая CNN ----------
class LectureSimpleCNN(nn.Module):
    """Conv(32,3x3,valid) -> MaxPool -> Conv(64,3x3,valid) -> MaxPool -> Dense(64) -> Dense(10)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(),     # 32x32 -> 30x30 (valid: (32-3)/1+1)
            nn.MaxPool2d(2),                    # 30 -> 15
            nn.Conv2d(32, 64, 3), nn.ReLU(),    # 15 -> 13
            nn.MaxPool2d(2),                    # 13 -> 6
        )
        self.head = nn.Sequential(
            nn.Flatten(),                       # 64 * 6 * 6 = 2304
            nn.Linear(64 * 6 * 6, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------- 3. Глубокая CNN (3 блока, BatchNorm, Dropout) ----------
def conv_bn(c_in, c_out, use_bn):
    """Conv(same) -> ReLU -> BatchNorm: порядок как в эталонном Keras-коде (activation внутри Conv2D)."""
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU()]
    if use_bn:
        layers.append(nn.BatchNorm2d(c_out))
    return layers


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (channel attention).

    GAP -> Dense(C/r, relu) -> Dense(C, sigmoid) -> маска (C,1,1) * вход.
    """

    def __init__(self, channels, ratio=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),            # squeeze: (B,C,H,W) -> (B,C)
            nn.Linear(channels, channels // ratio), nn.ReLU(),  # сжатие
            nn.Linear(channels // ratio, channels), nn.Sigmoid(),  # веса каналов 0..1
        )

    def forward(self, x):
        w = self.fc(x)[:, :, None, None]                      # (B,C) -> (B,C,1,1)
        return x * w                                          # excitation: перевзвешиваем каналы


class LectureDeepCNN(nn.Module):
    """Блоки 32-64-128 по две свёртки, MaxPool, Dropout 0.25/0.3/0.4, затем Dense(256)+BN+Dropout(0.5).

    Флаги use_bn / use_dropout / use_se / width нужны для ablation (по одному изменению за раз).
    """

    def __init__(self, num_classes=10, use_bn=True, use_dropout=True, use_se=False, width=32):
        super().__init__()
        w = [width, width * 2, width * 4]                     # 32-64-128 по умолчанию
        drops = [0.25, 0.3, 0.4]
        blocks, c_in = [], 3
        for c_out, p in zip(w, drops):
            layers = conv_bn(c_in, c_out, use_bn) + conv_bn(c_out, c_out, use_bn)
            if use_se:
                layers.append(SEBlock(c_out))
            layers.append(nn.MaxPool2d(2))                    # 32 -> 16 -> 8 -> 4
            if use_dropout:
                layers.append(nn.Dropout(p))
            blocks += layers
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        head = [nn.Flatten(), nn.Linear(w[2] * 4 * 4, 256), nn.ReLU()]
        if use_bn:
            head.append(nn.BatchNorm1d(256))
        if use_dropout:
            head.append(nn.Dropout(0.5))
        head.append(nn.Linear(256, num_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        return self.head(self.features(x))


# ---------- 4. CNN с attention ----------
class LectureSECNN(nn.Module):
    """Conv32 -> MaxPool -> SE -> Conv64 -> MaxPool -> SE -> Dense128 -> Dropout0.4 -> Dense10."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), SEBlock(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), SEBlock(64),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------- 5. Своя улучшенная модель: ResNet-9 (residual-связи + SE) ----------
def conv_block(c_in, c_out, pool=False):
    """Conv -> BN -> ReLU (современный порядок: нормализуем до нелинейности)."""
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    """y = x + F(x): градиент проходит по «короткому пути», глубокую сеть легче учить."""

    def __init__(self, c, use_se=True):
        super().__init__()
        self.body = nn.Sequential(conv_block(c, c), conv_block(c, c))
        self.se = SEBlock(c, ratio=16) if use_se else nn.Identity()

    def forward(self, x):
        return x + self.se(self.body(x))


class ResNet9(nn.Module):
    """width=64 — классический ResNet-9 (6.6M параметров); width=32 — в 4 раза дешевле по FLOPs."""

    def __init__(self, num_classes=10, use_se=True, dropout=0.2, width=64):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8
        self.net = nn.Sequential(
            conv_block(3, c1),                       # 32x32
            conv_block(c1, c2, pool=True),           # 16x16
            Residual(c2, use_se),
            conv_block(c2, c3, pool=True),           # 8x8
            conv_block(c3, c4, pool=True),           # 4x4
            Residual(c4, use_se),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),   # c4 признаков
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x):
        return self.net(x)


MODELS = {
    "mlp": LectureMLP,
    "simple_cnn": LectureSimpleCNN,
    "deep_cnn": LectureDeepCNN,
    "se_cnn": LectureSECNN,
    "resnet9": ResNet9,
}


def build_model(arch, num_classes=10, **kwargs):
    return MODELS[arch](num_classes=num_classes, **kwargs)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
