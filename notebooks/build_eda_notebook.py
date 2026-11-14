"""Build and execute notebooks/eda_hw2.ipynb.

NOTE: markdown polished by hand in eda_hw2.ipynb.
If you only need to refresh outputs, execute the existing notebook instead of rebuilding from this script.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "eda_hw2.ipynb"
FIG_DIR = ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def md(source: str):
    return nbf.v4.new_markdown_cell(source)


def code(source: str):
    return nbf.v4.new_code_cell(source)


cells = [
    md(
        """# EDA для Feedback Assistant (ДЗ2)

Исследовательский анализ данных для проекта ИТМО: ИИ-ассистент обратной связи по работам студентов.

**Цель EDA:** понять состав, объём и качество доступных данных до обучения и прогона моделей. Постановка задачи и бизнес-требования уже описаны в `docs/ml_system_design_doc.md` (ДЗ1) — здесь только данные и выводы.

**Источники:** срез из командного репозитория `task-reviewer` (`legacy/Архив`, `Тетрадь_*.md`, `РТ*.pdf`). Подробности — в `data/README.md`.
"""
    ),
    md("## 0. Подготовка"),
    code(
        """from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (9, 4.5)
plt.rcParams["axes.titlesize"] = 12

ROOT = Path("..").resolve()
if not (ROOT / "data" / "raw").exists():
    ROOT = Path(".").resolve()

DATA = ROOT / "data" / "raw"
FIG = ROOT / "notebooks" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

print("DATA =", DATA)
print("exists:", DATA.exists())
"""
    ),
    md(
        """## 1. Состав и объём данных

Смотрим, какие сущности есть для пайплайна «задание + рубрика + работа → черновик фидбэка».
"""
    ),
    code(
        """archive = DATA / "archive"
ru_dir = DATA / "ru_feedback"
pdf_dir = DATA / "pdf"
pdf_text_dir = DATA / "pdf_text"

tasks = [
    json.loads(line)
    for line in (archive / "tasks_parsed.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
checks_scored = [
    json.loads(line)
    for line in (archive / "checks_scored.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
checks_parsed = [
    json.loads(line)
    for line in (archive / "checks_parsed.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]

ru_files = sorted(ru_dir.glob("*.md"))
pdf_files = sorted(pdf_dir.glob("*.pdf"))
pdf_text_files = sorted(pdf_text_dir.glob("*.txt"))

inventory = pd.DataFrame(
    [
        {
            "сущность": "Задания + рубрики",
            "путь": "data/raw/archive/tasks_parsed.jsonl",
            "формат": "jsonl",
            "объём": len(tasks),
        },
        {
            "сущность": "Размеченные работы (scored)",
            "путь": "data/raw/archive/checks_scored.jsonl",
            "формат": "jsonl",
            "объём": len(checks_scored),
        },
        {
            "сущность": "Работы с исходной разметкой (parsed)",
            "путь": "data/raw/archive/checks_parsed.jsonl",
            "формат": "jsonl",
            "объём": len(checks_parsed),
        },
        {
            "сущность": "Русские тексты (тетради)",
            "путь": "data/raw/ru_feedback/",
            "формат": "md",
            "объём": len(ru_files),
        },
        {
            "сущность": "PDF работ / метрика",
            "путь": "data/raw/pdf/",
            "формат": "pdf",
            "объём": len(pdf_files),
        },
        {
            "сущность": "Извлечённый текст PDF",
            "путь": "data/raw/pdf_text/",
            "формат": "txt",
            "объём": len(pdf_text_files),
        },
    ]
)
inventory
"""
    ),
    code(
        """print("Задания:")
for t in tasks:
    print(
        f"- {t['id']}: критериев = {len(t['criteria'])}, "
        f"длина формулировки = {len(t['task'].split())} слов"
    )

print("\\nКритерии первого задания:")
for name, weight in tasks[0]["criteria"]:
    print(f"  [{weight}] {name}")
"""
    ),
    md(
        """**Вывод по составу.** Для бейзлайна есть полный контур: задание, рубрика (около 10 бинарных критериев) и 8 размеченных английских работ с фидбэком. Для русского MVP есть 27 текстовых файлов и несколько PDF, но **без той же покритериальной разметки 0/1**. Объём размеченных баллов маленький — для обучения классической модели мало, для EDA и оффлайн-оценки бейзлайна достаточно как стартового демо-набора.
"""
    ),
    md(
        """## 2. Качество размеченного набора (английские эссе)

Проверяем длины ответов, баланс меток по критериям и полноту полей.
"""
    ),
    code(
        """rows = []
for item in checks_scored:
    answer = item.get("answer", "")
    scores = item.get("criteria_results") or item.get("criteria_result") or []
    rows.append(
        {
            "task_id": item["task_id"],
            "n_words": len(answer.split()),
            "n_chars": len(answer),
            "n_criteria": len(scores),
            "n_positive": sum(int(s) for _, s in scores),
            "has_feedback": bool(str(item.get("feedback", "")).strip()),
            "feedback_words": len(str(item.get("feedback", "")).split()),
        }
    )

scored_df = pd.DataFrame(rows)
scored_df
"""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(scored_df["n_words"], bins=8, ax=axes[0], color="#4C78A8")
axes[0].set_title("Длина ответа студента (слова)")
axes[0].set_xlabel("слов в answer")

sns.barplot(data=scored_df, x="task_id", y="n_words", ax=axes[1], color="#F58518", errorbar=None)
axes[1].set_title("Длина ответа по заданию")
axes[1].tick_params(axis="x", rotation=20)
axes[1].set_xlabel("")
plt.tight_layout()
fig.savefig(FIG / "scored_answer_lengths.png", dpi=140, bbox_inches="tight")
plt.show()

print(scored_df[["n_words", "n_positive", "feedback_words"]].describe().round(1))
print("\\nПустые feedback:", int((~scored_df["has_feedback"]).sum()))
print("Работ по заданиям:\\n", scored_df["task_id"].value_counts().to_string())
"""
    ),
    md(
        """**Вывод.** Длины ответов сильно различаются: есть и короткие (~45 слов), и нормальные эссе (~250 слов). Пустых фидбэков нет. Работ всего 8 на 2 задания — выборка узкая, выводы по метрикам будут шумными.
"""
    ),
    code(
        """crit_rows = []
for item in checks_scored:
    scores = item.get("criteria_results") or item.get("criteria_result") or []
    for name, score in scores:
        crit_rows.append({"task_id": item["task_id"], "criterion": name, "score": int(score)})

crit_df = pd.DataFrame(crit_rows)
balance = crit_df.groupby("criterion")["score"].agg(["mean", "sum", "count"]).reset_index()
balance = balance.sort_values("mean")
balance["negative"] = balance["count"] - balance["sum"]
balance
"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(10, 5))
plot_df = balance.sort_values("mean")
ax.barh(plot_df["criterion"], plot_df["mean"], color="#54A24B")
ax.set_xlim(0, 1)
ax.set_xlabel("доля выполненных (score=1)")
ax.set_title("Баланс меток по критериям (checks_scored)")
plt.tight_layout()
fig.savefig(FIG / "criteria_positive_rate.png", dpi=140, bbox_inches="tight")
plt.show()

print("Всего меток:", len(crit_df))
print("score=1:", int((crit_df["score"] == 1).sum()), "score=0:", int((crit_df["score"] == 0).sum()))
print("Критерии почти всегда выполнены (mean >= 0.85):")
print(balance.loc[balance["mean"] >= 0.85, ["criterion", "mean"]].to_string(index=False))
"""
    ),
    md(
        """**Вывод по качеству разметки.** Метки бинарные (0/1), есть и положительные, и отрицательные примеры, но по части критериев почти нет негативных случаев. Для F1 это риск: модель может «угадывать единицу». Для бейзлайна набор годится, для устойчивого обучения нужно больше работ и более равномерная разметка.
"""
    ),
    code(
        """parsed_map = {}
for item in checks_parsed:
    key = (item["task_id"], item["answer"][:80])
    parsed_map[key] = {name: int(s) for name, s in item.get("criteria_results", [])}

diffs = []
for item in checks_scored:
    key = (item["task_id"], item["answer"][:80])
    scored = {name: int(s) for name, s in (item.get("criteria_results") or [])}
    parsed = parsed_map.get(key)
    if not parsed:
        continue
    for name in scored:
        if name in parsed and parsed[name] != scored[name]:
            diffs.append(
                {
                    "task_id": item["task_id"],
                    "criterion": name,
                    "parsed": parsed[name],
                    "scored": scored[name],
                }
            )

diff_df = pd.DataFrame(diffs)
matched = sum(1 for item in checks_scored if (item["task_id"], item["answer"][:80]) in parsed_map)
print("Пар работ matched:", matched)
print("Расхождений parsed vs scored:", len(diff_df))
diff_df.head(10)
"""
    ),
    md(
        """**Вывод.** Между `checks_parsed` и `checks_scored` есть расхождения по критериям. Для валидации нужно явно зафиксировать ground truth. Дальше в этом ноутбуке используем `checks_scored.jsonl`.
"""
    ),
    md(
        """## 3. Русский корпус (`Тетрадь_*.md`)

Файлы выглядят как текстовые разборы и черновики обратной связи по нескольким заданиям. Для MVP на русском важны длина, язык и риск персональных данных.
"""
    ),
    code(
        r"""def load_ru_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if "output_preview" in text[:100]:
        match = re.search(r'"output_preview"\s*:\s*"(.*)"\s*$', text, flags=re.S)
        if match:
            raw = match.group(1)
            try:
                return bytes(raw, "utf-8").decode("unicode_escape")
            except Exception:
                return raw.replace("\\n", "\n").replace('\\"', '"')
    return text


ru_rows = []
email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
phone_re = re.compile(r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}")
cyr_re = re.compile(r"[а-яА-ЯёЁ]")
lat_re = re.compile(r"[A-Za-z]")

for path in ru_files:
    text = load_ru_text(path)
    cyr = len(cyr_re.findall(text))
    lat = len(lat_re.findall(text))
    ru_rows.append(
        {
            "file": path.name,
            "n_chars": len(text),
            "n_words": len(text.split()),
            "cyr_share": cyr / max(cyr + lat, 1),
            "n_emails": len(email_re.findall(text)),
            "n_phones": len(phone_re.findall(text)),
            "has_assignment_markers": int("ЗАДАНИЕ" in text.upper() or "Задание" in text),
        }
    )

ru_df = pd.DataFrame(ru_rows).sort_values("n_words")
ru_df.head()
"""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(ru_df["n_words"], bins=10, ax=axes[0], color="#4C78A8")
axes[0].set_title("Длина русских текстов (слова)")
axes[0].set_xlabel("слов")

sns.boxplot(y=ru_df["n_words"], ax=axes[1], color="#72B7B2")
axes[1].set_title("Разброс длин")
axes[1].set_ylabel("слов")
plt.tight_layout()
fig.savefig(FIG / "ru_lengths.png", dpi=140, bbox_inches="tight")
plt.show()

print(ru_df[["n_words", "n_chars", "cyr_share"]].describe().round(3))
print("\\nДоля файлов с кириллицей > 70%:", float((ru_df["cyr_share"] > 0.7).mean().round(2)))
print("Файлы с email:", int((ru_df["n_emails"] > 0).sum()))
print("Файлы с телефонами:", int((ru_df["n_phones"] > 0).sum()))
print("Файлы с маркером ЗАДАНИЕ:", int(ru_df["has_assignment_markers"].sum()))
"""
    ),
    md(
        """**Вывод.** Русский набор заметно больше размеченного английского по числу документов, тексты в основном на кириллице, email и телефоны простым поиском не найдены. Но это **не размеченные студенческие эссе с баллами по рубрике**, а скорее готовые разборы. Для генерации русского фидбэка полезно как примеры стиля; для MAE/F1 по критериям — недостаточно.
"""
    ),
    md(
        """## 4. PDF: извлекаемость текста

Проверяем, получается ли достать текст без OCR. В скоупе MVP OCR нет.
"""
    ),
    code(
        """pdf_rows = []
for pdf_path in pdf_files:
    txt_path = pdf_text_dir / f"{pdf_path.stem}.txt"
    text = txt_path.read_text(encoding="utf-8", errors="replace") if txt_path.exists() else ""
    letters = sum(ch.isalnum() for ch in text)
    pdf_rows.append(
        {
            "file": pdf_path.name,
            "size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
            "extracted_chars": len(text),
            "alnum_chars": letters,
            "extracted_words": len(text.split()),
            "looks_empty": letters < 200,
            "kind": "metric" if "Метрика" in pdf_path.name else "submission",
        }
    )

pdf_df = pd.DataFrame(pdf_rows).sort_values("extracted_words")
pdf_df
"""
    ),
    code(
        """plot_pdf = pdf_df[pdf_df["kind"] == "submission"].copy()
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=plot_pdf, x="file", y="extracted_words", color="#E45756", ax=ax)
ax.set_title("Слов после pdftotext по PDF работ")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
fig.savefig(FIG / "pdf_extract_words.png", dpi=140, bbox_inches="tight")
plt.show()

print("Пустых/почти пустых PDF:", int(pdf_df["looks_empty"].sum()))
print(pdf_df[["file", "size_mb", "extracted_words", "looks_empty"]].to_string(index=False))
"""
    ),
    md(
        """**Вывод.** У доступных `РТ*.pdf` текст извлекается (тысячи слов), то есть это не чистые сканы. Файлы тяжёлые (по несколько МБ) — внутри, вероятно, есть изображения и оформление. Для MVP парсинг PDF нужен; OCR можно отложить, но нужен явный fallback, если текст не извлечётся.
"""
    ),
    md(
        """## 5. Связь с метриками качества

Считаем простые статистики по критериям на `checks_scored`. Полный прогон LLM здесь не делаем. В ДЗ1 уже выбраны MAE, Precision, Recall, F1 и HITL-метрики.
"""
    ),
    code(
        """tmp = []
for i, item in enumerate(checks_scored):
    scores = item.get("criteria_results") or []
    total = sum(int(s) for _, s in scores)
    tmp.append(
        {
            "work_idx": i,
            "task_id": item["task_id"],
            "total_score": total,
            "max_score": len(scores),
        }
    )
work_df = pd.DataFrame(tmp)
work_df["share"] = work_df["total_score"] / work_df["max_score"]
work_df
"""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(
    work_df["total_score"],
    bins=range(0, int(work_df["max_score"].max()) + 2),
    ax=axes[0],
    color="#B279A2",
)
axes[0].set_title("Суммарный балл работы")
axes[0].set_xlabel("число критериев со score=1")

axes[1].bar(
    ["score=0", "score=1"],
    [(crit_df["score"] == 0).mean(), (crit_df["score"] == 1).mean()],
    color=["#E45756", "#54A24B"],
)
axes[1].set_ylim(0, 1)
axes[1].set_title("Глобальный баланс меток")
axes[1].set_ylabel("доля")
plt.tight_layout()
fig.savefig(FIG / "score_distribution.png", dpi=140, bbox_inches="tight")
plt.show()

print(work_df[["total_score", "share"]].describe().round(3))
print(f"Глобальная доля положительных меток: {crit_df['score'].mean():.3f}")
"""
    ),
    md(
        """**Вывод по метрикам.** На текущих данных естественно считать покритериальные Precision/Recall/F1 и MAE по сумме баллов. Ориентир легаси MAE 1.66 и F1 0.86 относится к более широкой апробации. На 8 работах этого среза можно воспроизвести протокол оценки, но нельзя заявлять те же цифры как новый результат. Для русского корпуса сначала нужна экспертная разметка части тетрадей или PDF.
"""
    ),
    md(
        """## 6. Главные выводы

1. **Состав данных достаточный для EDA и демо-бейзлайна**, но не для устойчивого обучения: 2 задания, 8 размеченных EN-работ, 27 RU-текстов, 5 PDF работ.
2. **Размеченный контур полный** (task + criteria + answer + scores + feedback) — главный актив для этапа моделей.
3. **Баланс критериев неравномерный** — часть критериев почти всегда равна 1, F1 будет смещён.
4. **Русский набор больше по объёму**, но без покритериальных баллов — риск для заявленного русского MVP.
5. **PDF-текст извлекается** без OCR на текущей выборке; тяжёлые файлы и риск сканов всё равно надо учитывать.
6. **PII простым поиском не найдены** (email/телефон), но маскирование перед LLM всё равно обязательно.
7. **Нехватка данных:** для уверенной оценки бейзлайна желательно порядка 50–100 размеченных работ на нескольких рубриках и 20–30 русских работ с экспертными баллами.
"""
    ),
    md(
        """## 7. Что это значит для следующих этапов

- **Подготовка данных:** зафиксировать `checks_scored` как ground truth для EN-бейзлайна; русские md/PDF держать отдельно как unlabeled / style corpus.
- **Модели:** валидацию делать по `task_id` и работе целиком; метрики — MAE, F1 и HITL-принятие черновика.
- **Специфичные этапы:** парсинг PDF, PII-фильтр и structured output под рубрику обязательны из-за картины данных выше.

Описание этапов будет дописано в `docs/ml_system_design_doc.md` (раздел 2.3) в рамках ДЗ2.
"""
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.cells = cells
    nbf.write(nb, NB_PATH)
    print("wrote", NB_PATH)

    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_PATH.parent)}},
    )
    client.execute()
    nbf.write(nb, NB_PATH)
    print("executed", NB_PATH)
    print("figures", sorted(p.name for p in FIG_DIR.glob("*.png")))


if __name__ == "__main__":
    main()
