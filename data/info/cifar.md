# CIFAR-10 и CIFAR-100

- **Авторы:** Alex Krizhevsky, Vinod Nair, Geoffrey Hinton (2009), подмножество 80 Million Tiny Images.
- **Оригинал:** https://www.cs.toronto.edu/~kriz/cifar.html (качается очень медленно).
- **Что лежит в `data/raw/`:** parquet-зеркала Hugging Face `uoft-cs/cifar10` и `uoft-cs/cifar100`, содержимое то же, что в оригинале. Колонки: `img` (PNG-байты), `label` (CIFAR-10) или `fine_label` + `coarse_label` (CIFAR-100).
- **Один пример:** цветная картинка 32×32×3, uint8 0…255, и номер класса.

| | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Классов | 10 | 100 (20 суперклассов по 5) |
| Train | 50 000 (5 000 на класс) | 50 000 (500 на класс) |
| Test | 10 000 (1 000 на класс) | 10 000 (100 на класс) |

Классы CIFAR-10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Классы не пересекаются (automobile это легковые, truck только большие грузовики).

`data/processed/cifar10.npz` создаётся кодом (`src/data.py`, `load_cifar`) из parquet: декодированные массивы для быстрой загрузки.
