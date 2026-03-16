# Training Interface для Нормализации Навыков

Этот проект предоставляет интерфейс для обучения модели на основе T5 для нормализации навыков, но при желании можно использовать для других задач. Модель обучается на парах входных и выходных текстов, где вход - сырой навык, а выход - нормализованная версия.

## Описание

Класс `Normalize_Model` реализует обучение модели T5 для задачи нормализации навыков. Поддерживает аугментацию данных, раннюю остановку, тестирование с кросс-валидацией и визуализацию результатов обучения.

## Установка

Убедитесь, что у вас установлен Python 3.7+.

Установите необходимые зависимости:

```bash
pip install -r requirements.txt
```

## Использование

### Инициализация модели

```python
from train import Normalize_Model

model = Normalize_Model(
    batch_size=16,
    epochs=16,
    lr=2e-4,
    weight_decay=0.01,
    patience=3,
    min_delta=0.01,
    task="normalize skill",
    start_model="./start_model"
)
```

### Обучение модели

```python
pairs = [
    ("питон", "Python"),
    ("ml", "Machine Learning"),
    # ... ваши данные
]

model.train(pairs, test_pairs=test_pairs, aug=True)
```

### Тестирование модели

```python
accuracy = model.test(pairs, branches=5, aug=True)
print(f"Accuracy: {accuracy}")
```

### Генерация ответа

```python
normalized_skill = model.answer("python prog")
print(normalized_skill)  # "Python Programming"
```

### Сохранение модели

```python
model.save("new_model")
```

### Построение графиков

```python
model.graph()
```

## Параметры

- `batch_size`: Размер батча для обучения (по умолчанию 16)
- `epochs`: Количество эпох обучения (по умолчанию 16)
- `lr`: Скорость обучения (по умолчанию 2e-4)
- `weight_decay`: Регуляризация (по умолчанию 0.01)
- `patience`: Терпение для ранней остановки (по умолчанию 3)
- `min_delta`: Минимальное изменение для ранней остановки (по умолчанию 0.01)
- `task`: Префикс задачи (по умолчанию "normalize skill")
- `start_model`: Путь к начальной модели (по умолчанию "./start_model")

## Аугментация данных

Метод `augmentation` добавляет шум к данным для улучшения обобщающей способности модели:
- Добавление случайных символов
- Удаление символов
- Замена символов
- Изменение регистра

## Тестирование

Метод `test` использует k-fold кросс-валидацию для оценки модели.

## Зависимости

- PyTorch
- Transformers
- NumPy
- Matplotlib
