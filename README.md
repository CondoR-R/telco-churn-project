```text
telco-churn-project/
│
├── data/
│   ├── raw/                    # исходный CSV с Kaggle, не изменяется вручную
│   └── processed/              # результат предобработки (train/test выборки)
│
├── notebooks/
│   ├── 01_eda.ipynb            # весь анализ из плана EDA
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb       # сравнение моделей, подбор гиперпараметров
│
├── src/
│   ├── ml/
│   │   ├── preprocessing.py    # пайплайн очистки + фичей (используется и в notebooks, и в сервисе)
│   │   ├── train.py            # обучение финальной модели, сохранение артефакта
│   │   └── model.py            # обёртка загрузки модели для инференса
│   │
│   ├── api/
│   │   ├── main.py             # точка входа FastAPI, роуты
│   │   ├── schemas.py          # pydantic-модели запроса/ответа
│   │   └── dependencies.py     # инициализация модели, БД-сессии и т.п.
│   │
│   └── db/
│       ├── models.py           # SQLAlchemy-модели (таблица логов запросов)
│       ├── database.py         # подключение к SQLite, сессии
│       └── crud.py             # функции записи/чтения логов
│
├── models/
│   └── churn_model.pkl         # сериализованный финальный pipeline (предобработка + модель)
│
├── tests/
│   ├── test_api.py             # тесты эндпоинтов (валидация, формат ответа)
│   ├── test_model.py           # тест качества модели на фиксированных примерах
│   └── test_preprocessing.py   # тесты корректности предобработки данных
│
├── Dockerfile                  # сборка образа сервиса
├── .dockerignore
├── requirements.txt            # или pyproject.toml, если предпочитаете poetry
├── .env.example                # пример переменных окружения (путь к БД, к модели и т.п.)
├── .gitignore
└── README.md                   # описание проекта, как запустить, метрики модели
```
