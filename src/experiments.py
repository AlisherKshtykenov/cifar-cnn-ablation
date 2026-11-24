"""Все эксперименты. Каждый ablation-эксперимент меняет ровно одну вещь относительно предыдущего.

Запуск всех (долго, в фоне):  python3 -m src.experiments
"""
LECTURE = [  # 1. эталонные архитектуры как есть: Adam lr=1e-3, batch 128, без аугментации
    {"name": "e1_mlp", "arch": "mlp", "optimizer": "adam", "epochs": 20},
    {"name": "e2_simple_cnn", "arch": "simple_cnn", "optimizer": "adam", "epochs": 20},
    {"name": "e3_deep_cnn", "arch": "deep_cnn", "optimizer": "adam", "epochs": 20},
    {"name": "e4_se_cnn", "arch": "se_cnn", "optimizer": "adam", "epochs": 20},
]

ABLATION = [  # 2. глубокая CNN: что даёт каждая техника
    {"name": "a1_deep_plain", "arch": "deep_cnn", "optimizer": "adam", "epochs": 20,
     "model_kwargs": {"use_bn": False, "use_dropout": False}},
    {"name": "a2_deep_bn", "arch": "deep_cnn", "optimizer": "adam", "epochs": 20,
     "model_kwargs": {"use_bn": True, "use_dropout": False}},
    {"name": "a3_deep_dropout", "arch": "deep_cnn", "optimizer": "adam", "epochs": 20,
     "model_kwargs": {"use_bn": False, "use_dropout": True}},
    # BN + Dropout = e3_deep_cnn
    {"name": "a4_deep_aug", "arch": "deep_cnn", "optimizer": "adam", "epochs": 20, "augment": True},
    {"name": "a5_deep_aug_se", "arch": "deep_cnn", "optimizer": "adam", "epochs": 20, "augment": True,
     "model_kwargs": {"use_se": True}},
    {"name": "a6_deep_aug_onecycle", "arch": "deep_cnn", "optimizer": "sgd_onecycle", "lr": 0.05,
     "epochs": 20, "augment": True},
    {"name": "a7_deep_wide_aug_onecycle", "arch": "deep_cnn", "optimizer": "sgd_onecycle", "lr": 0.05,
     "epochs": 20, "augment": True, "model_kwargs": {"width": 64}},
]

FINAL = [  # 3. своя модель: ResNet-9 + SE + аугментация + OneCycle + label smoothing
    {"name": "f1_resnet9", "arch": "resnet9", "optimizer": "sgd_onecycle", "lr": 0.1, "wd": 5e-4,
     "epochs": 30, "augment": True, "label_smoothing": 0.1, "model_kwargs": {"width": 32}},
]

ALL = LECTURE + ABLATION + FINAL
BY_NAME = {c["name"]: c for c in ALL}

if __name__ == "__main__":
    import sys

    from .train import prepare_data, run_experiment

    names = sys.argv[1:] or [c["name"] for c in LECTURE + ABLATION[:-1] + FINAL + ABLATION[-1:]]
    data = prepare_data("cifar10")
    for name in names:
        _, _, s = run_experiment(BY_NAME[name], data)
        print("DONE", s, flush=True)
