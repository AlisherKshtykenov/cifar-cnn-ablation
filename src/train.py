"""Цикл обучения CNN на GPU (Apple MPS / CUDA / CPU) с аугментацией прямо на устройстве и кэшем результатов.

run_experiment(cfg) обучает модель один раз, сохраняет веса в weights/ и историю в results/logs/.
При повторном вызове с тем же именем просто читает кэш: ноутбук перезапускается за минуты.
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import ROOT, channel_stats, load_cifar, split_train_val
from .models import build_model, count_params

WEIGHTS = ROOT / "weights"
LOGS = ROOT / "results" / "logs"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensor(x_uint8, mean, std, device):
    """uint8 (N,H,W,C) -> float (N,C,H,W), нормализация по каналам: (x/255 - mean) / std."""
    x = torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float().div_(255)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s


def augment(x, pad=4):
    """Random crop 32x32 из картинки с отражённой рамкой 4 px + случайное отражение по горизонтали."""
    b, c, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")              # (B,C,40,40)
    dy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)          # случайный сдвиг каждой картинки
    dx = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = (dy[:, None] + torch.arange(h, device=x.device))[:, :, None]   # (B,H,1)
    cols = (dx[:, None] + torch.arange(w, device=x.device))[:, None, :]   # (B,1,W)
    bidx = torch.arange(b, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[bidx, rows, cols].permute(0, 3, 1, 2)  # вырезали окна
    flip = (torch.rand(b, 1, 1, 1, device=x.device) < 0.5)
    out = torch.where(flip, out.flip(3), out)                          # отражение половины батча (без синхронизации GPU)
    return out.contiguous()


@torch.no_grad()
def predict_logits(model, x, batch_size=1000):
    """Предсказание в режиме eval: Dropout выключен, BatchNorm берёт накопленную статистику."""
    model.eval()
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def evaluate(model, x, y, criterion):
    logits = predict_logits(model, x)
    loss = criterion(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


def prepare_data(dataset="cifar10", device=None, seed=42):
    """Загрузка, разбиение 45k/5k/10k, нормализация статистикой train. Всё сразу на GPU (~0.7 ГБ)."""
    device = device or get_device()
    x_tr, y_tr, x_te, y_te = load_cifar(dataset)
    x_tr, x_va, y_tr, y_va = split_train_val(x_tr, y_tr, seed=seed)
    mean, std = channel_stats(x_tr)
    t = lambda x: to_tensor(x, mean, std, device)
    l = lambda y: torch.from_numpy(y).to(device)
    return {"x_train": t(x_tr), "y_train": l(y_tr), "x_val": t(x_va), "y_val": l(y_va),
            "x_test": t(x_te), "y_test": l(y_te), "mean": mean, "std": std,
            "num_classes": int(y_tr.max()) + 1}


def make_optimizer(model, cfg, steps_per_epoch):
    """'adam' = как в эталоне (Adam, lr=1e-3 постоянный); 'onecycle' = SGD Nesterov + OneCycle."""
    if cfg["optimizer"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
        return opt, None
    if cfg["optimizer"] == "adamw_onecycle":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-3), weight_decay=cfg.get("wd", 5e-4))
    else:  # sgd_onecycle
        opt = torch.optim.SGD(model.parameters(), lr=cfg.get("lr", 0.1), momentum=0.9, nesterov=True,
                              weight_decay=cfg.get("wd", 5e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 0.1), epochs=cfg["epochs"],
                                                steps_per_epoch=steps_per_epoch, pct_start=0.25)
    return opt, sched


def train(cfg, data, verbose=True):
    """Mini-batch обучение. Лучшая эпоха выбирается по val accuracy, тест считается один раз в конце."""
    torch.manual_seed(cfg.get("seed", 42))
    device = data["x_train"].device
    model = build_model(cfg["arch"], data["num_classes"], **cfg.get("model_kwargs", {})).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    eval_criterion = nn.CrossEntropyLoss()  # val/test loss всегда без сглаживания, чтобы сравнивать модели
    bs, n = cfg.get("batch_size", 128), len(data["y_train"])
    steps = (n + bs - 1) // bs
    opt, sched = make_optimizer(model, cfg, steps)

    history, best_acc, best_state, best_epoch = [], -1.0, None, 0
    start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.perf_counter()
        order = torch.randperm(n, device=device)              # новый порядок объектов каждую эпоху
        run_loss = torch.zeros((), device=device)                # копим на GPU: .item() на каждом шаге
        run_correct = torch.zeros((), device=device)             # заставлял бы CPU ждать GPU после каждого батча
        for i in range(steps):
            idx = order[i * bs:(i + 1) * bs]
            xb, yb = data["x_train"][idx], data["y_train"][idx]
            if cfg.get("augment", False):
                xb = augment(xb)
            logits = model(xb)                                # прямой проход
            loss = criterion(logits, yb)                      # cross-entropy
            opt.zero_grad(set_to_none=True)                   # обнулить старые градиенты
            loss.backward()                                   # backpropagation
            opt.step()                                        # шаг оптимизатора
            if sched is not None:
                sched.step()                                  # OneCycle меняет lr каждый батч
            run_loss += loss.detach() * len(yb)
            run_correct += (logits.argmax(1) == yb).sum()
        train_loss_eval, train_acc_eval = run_loss.item() / n, run_correct.item() / n
        val_loss, val_acc = evaluate(model, data["x_val"], data["y_val"], eval_criterion)
        if device.type == "mps":
            torch.mps.synchronize()
        history.append({"epoch": epoch, "train_loss": train_loss_eval, "train_acc": train_acc_eval,
                        "val_loss": val_loss, "val_acc": val_acc, "sec": time.perf_counter() - t0})
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"[{cfg['name']}] ep {epoch:3d} train {train_loss_eval:.3f}/{train_acc_eval:.3f} "
                  f"val {val_loss:.3f}/{val_acc:.3f} {history[-1]['sec']:.1f}s", flush=True)

    model.load_state_dict(best_state)                         # берём лучшую по val эпоху
    test_loss, test_acc = evaluate(model, data["x_test"], data["y_test"], eval_criterion)
    summary = {"name": cfg["name"], "params": count_params(model), "best_epoch": best_epoch,
               "best_val_acc": best_acc, "final_train_acc": history[-1]["train_acc"],
               "final_val_acc": history[-1]["val_acc"],
               "min_val_loss": min(h["val_loss"] for h in history),
               "test_loss": test_loss, "test_acc": test_acc,
               "train_minutes": (time.perf_counter() - start) / 60, "device": device.type}
    return model, history, summary


def run_experiment(cfg, data=None, force=False, verbose=True):
    """Обучает или берёт из кэша. Возвращает (model, history, summary)."""
    WEIGHTS.mkdir(exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    w_path, log_path = WEIGHTS / f"{cfg['name']}.pt", LOGS / f"{cfg['name']}.json"
    if log_path.exists() and w_path.exists() and not force:
        rec = json.loads(log_path.read_text())
        device = data["x_train"].device if data is not None else get_device()
        model = build_model(cfg["arch"], rec["num_classes"], **cfg.get("model_kwargs", {})).to(device)
        model.load_state_dict(torch.load(w_path, map_location=device))
        return model, rec["history"], rec["summary"]
    data = data or prepare_data(cfg.get("dataset", "cifar10"))
    model, history, summary = train(cfg, data, verbose=verbose)
    torch.save(model.state_dict(), w_path)
    log_path.write_text(json.dumps({"config": cfg, "num_classes": data["num_classes"],
                                    "history": history, "summary": summary}, indent=1))
    return model, history, summary
