# Данные для EDA (ДЗ2)

Это кусок данных из командного репозитория `task-reviewer`. Email и телефоны простым поиском не нашлись. Перед реальной работой всё равно нужна ручная проверка на имена и другие личные данные.

## Что где лежит

| Путь | Что внутри | Сколько |
| --- | --- | --- |
| `raw/archive/` | задания, рубрики и английские эссе с баллами | 2 задания, 8 работ, около 10 критериев |
| `raw/ru_feedback/` | русские `Тетрадь_*.md` | 27 файлов |
| `raw/pdf/` | PDF работ и файл с метрикой фидбэка | 5 РТ + Метрика_done |
| `raw/pdf_text/` | текст, вытащенный из PDF | 6 txt |

## Откуда взяли

- `archive/` — из `legacy/Архив`
- `ru_feedback/` — из `legacy/Тетрадь_*.md`
- `pdf/` — из `legacy/РТ*.pdf` и `Метрика_done.pdf`

Разбор: `notebooks/eda_hw2.ipynb`.

## Как снова запустить ноутбук

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
MPLBACKEND=Agg python notebooks/build_eda_notebook.py
```

Или просто открой `notebooks/eda_hw2.ipynb` и прогони ячейки сверху вниз.
