"""
Кандидатогенерация для задачи Авито.
Разработка решения, которое для каждого поискового запроса отбирает из корпуса до 50 объявлений кандидатов,
которые дальше уйдут на ранжирование.
Принимаются на вход и читаются train.parquet, benchmark_queries.parquet, benchmark_items.parquet.
Записывается answer.csv с колонками query_id,answer.
"""

import numpy as np
import pandas as pd
from scipy import sparse
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


# 1. Прописываем пути к файлам
TRAIN_PATH = "train.parquet"
QUERIES_PATH = "benchmark_queries.parquet"
ITEMS_PATH = "benchmark_items.parquet"
OUTPUT_PATH = "answer.csv"


# 2. Загружаем, считываем данные
train = pd.read_parquet(TRAIN_PATH)
queries = pd.read_parquet(QUERIES_PATH)
items = pd.read_parquet(ITEMS_PATH)

# id приводим к строке (иначе теряются ведущие нули и регистр)
for df, col in [(queries, "query_id"), (items, "item_id"), (train, "item_id")]:
    if col in df.columns:
        df[col] = df[col].astype(str)


# 3. Нормализация текста

def normalize_text(s: str) -> str:
    """Функция приводит текст к нижнему регистру, убирает лишние символы."""
    if pd.isna(s):
        return ""
    s = str(s).lower().replace("ё", "е")
    # оставляем буквы, цифры и пробелы
    s = re.sub(r"[^0-9a-zа-я]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_item_text(df: pd.DataFrame) -> pd.Series:
    """Собирает единый текст объявления.
    Заголовок повторяем дважды, чтобы он имел больший вес по tf-idf."""
    title = df["item_title_raw"].fillna("").astype(str)
    desc = df["item_description_raw"].fillna("").astype(str)
    params = df["item_infm_params_text"].fillna("").astype(str)

    parts = [
        title + " " + title,
        desc,
        params,
    ]
    
    # превращаем категории и подкатегории в токены, чтобы tf-idf мог учитывать их
    if "item_category_id" in df.columns:
        parts.append("cat_" + df["item_category_id"].astype(str))
    if "item_microcat_id" in df.columns:
        parts.append("mic_" + df["item_microcat_id"].astype(str))

    text = parts[0]
    for p in parts[1:]:
        text = text + " " + p

    return text.map(normalize_text)


def build_query_text(df: pd.DataFrame) -> pd.Series:
    """Собирает единый текст запроса.
    Сам search_query повторяем трижды, так как это главный сигнал запроса."""
    q = df["search_query"].fillna("").astype(str)
    
    parts = [
        q + " " + q + " " + q,
    ]

    if "search_infm_params_text" in df.columns:
        parts.append(df["search_infm_params_text"].fillna("").astype(str))
    if "search_category" in df.columns:
        parts.append("cat_" + df["search_category"].astype(str))

    text = parts[0]
    for p in parts[1:]:
        text = text + " " + p

    return text.map(normalize_text)

""" создаем новую колонку _text в датафреймах items и queries.
 В нее записывается результат работы функций build_item_text и build_query_text,
 то есть единый нормализованный текст для каждого объявления и каждого запроса."""
items["_text"] = build_item_text(items)
queries["_text"] = build_query_text(queries)


# 4. Полезные маппинги из train

# создаем словарь пар «категория запроса -> самая частая категория объявления».
cat_map = {}
if "search_category" in train.columns and "item_category_id" in train.columns:
    tmp = train[["search_category", "item_category_id"]].dropna().astype(str)
    for sc, group in tmp.groupby("search_category"):
        cat_map[sc] = Counter(group["item_category_id"]).most_common(1)[0][0]

# создаем словарь пар «локация запроса -> самая частая локация выбранного объявления».
loc_map = {}
if "search_location_id" in train.columns and "item_location_id" in train.columns:
    tmp = train[["search_location_id", "item_location_id"]].dropna().astype(str)
    for sl, group in tmp.groupby("search_location_id"):
        loc_map[sl] = Counter(group["item_location_id"]).most_common(1)[0][0]

# популярность объявления в train (сколько раз его выбирали).
item_pop = Counter(train["item_id"].astype(str)) if "item_id" in train.columns else Counter()


# 5. TF-IDF по объявлениям и запросам

vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    max_features=500_000,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
)

item_matrix = vectorizer.fit_transform(items["_text"]).tocsr()
query_matrix = vectorizer.transform(queries["_text"]).tocsr()

N = items.shape[0]
TOP_K = 50
BATCH_SIZE = 100


# 6. Массивы для бустов

item_ids = items["item_id"].astype(str).values

item_cat = items["item_category_id"].astype(str).values if "item_category_id" in items.columns else None
item_loc = items["item_location_id"].astype(str).values if "item_location_id" in items.columns else None

item_rating = (
    items["item_rating"].fillna(0).astype(float).values
    if "item_rating" in items.columns else None
)
item_reviews = (
    items["item_rating_reviews_count"].fillna(0).astype(float).values
    if "item_rating_reviews_count" in items.columns else None
)

# буст за популярность объявления в train (item_pop).
pop_boost = np.zeros(N, dtype=np.float32)
if item_pop:
    for i, iid in enumerate(item_ids):
        c = item_pop.get(iid, 0)
        if c > 0:
            pop_boost[i] = np.log1p(c)


# 7. Основной цикл retrieval + бусты

predictions = [[] for _ in range(len(queries))]

for start in range(0, len(queries), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(queries))
    q_batch = query_matrix[start:end]

    # shape: (batch, N)
    scores_batch = (q_batch @ item_matrix.T).toarray()

    for j in range(scores_batch.shape[0]):
        q_idx = start + j
        s = scores_batch[j].copy()

        max_s = float(s.max()) if s.size else 0.0

        # если текстовых совпадений нет, смотрим на популярность/рейтинг
        if max_s <= 0:
            s = pop_boost.copy()
            if item_rating is not None:
                s += 0.01 * np.nan_to_num(item_rating)
            max_s = float(s.max()) if s.size else 0.0

        qrow = queries.iloc[q_idx]

        # буст по категории
        if item_cat is not None and "search_category" in queries.columns:
            sc = str(qrow.get("search_category", ""))
            mapped = cat_map.get(sc, sc)
            if mapped:
                mask = item_cat == mapped
                s[mask] += 0.30 * max_s

        # буст по локации
        if item_loc is not None and "search_location_id" in queries.columns:
            sl = str(qrow.get("search_location_id", ""))
            mapped_loc = loc_map.get(sl, sl)
            if mapped_loc:
                mask = item_loc == mapped_loc
                s[mask] += 0.20 * max_s

        # буст за популярность в train
        if pop_boost.max() > 0:
            s += 0.05 * pop_boost

        # небольшие бусты за рейтинг и отзывы
        if item_rating is not None:
            s += 0.01 * np.nan_to_num(item_rating)
        if item_reviews is not None:
            s += 0.005 * np.log1p(np.nan_to_num(item_reviews))

        # берем top-50.
        if TOP_K >= len(s):
            idx = np.argsort(-s)[:TOP_K]
        else:
            idx = np.argpartition(-s, TOP_K)[:TOP_K]
            idx = idx[np.argsort(-s[idx])]

        predictions[q_idx] = [item_ids[i] for i in idx]


# 8. Сохраняем answer.csv

answer = pd.DataFrame({
    "query_id": queries["query_id"].astype(str),
    "answer": [" ".join(p) for p in predictions],
})

answer.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")


# 9. Проверка формата ответного файла

assert len(answer) == len(queries), "Число строк не совпадает с числом запросов"
assert answer["query_id"].is_unique, "query_id повторяются"

all_item_ids = set(item_ids)

for line in answer["answer"]:
    ids = line.split()
    assert len(ids) <= 50, "Больше 50 item_id в строке"
    assert len(ids) == len(set(ids)), "Повтор item_id внутри строки"
    for iid in ids:
        assert iid in all_item_ids, f"item_id {iid} отсутствует в benchmark_items"

print(f"Готово: {OUTPUT_PATH}")