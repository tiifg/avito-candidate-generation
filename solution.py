"""
Кандидатогенерация для поиска услуг: отбор до 50 объявлений на запрос.

Подход - гибридный лексический поиск с обучаемыми приорами:
  1) BM25 по тексту объявления, расширенному кликовыми запросами из train;
  2) символьный TF-IDF (char 3-4) - устойчивость к опечаткам и морфологии;
  3) collaborative kNN: похожие train-запросы -> выбранные по ним объявления;
  4) приоры по микрокатегории / категории / локации, обученные на train;
  5) слабые приоры популярности и рейтинга как тай-брейк.

Скоры всех источников нормируются на максимум внутри запроса и складываются
с весами W.

Зависимости (все локальные, без внешних API): numpy, pandas, scipy,
scikit-learn, nltk (SnowballStemmer для русского).
"""

import os
import pickle
import re
import time
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from nltk.stem.snowball import SnowballStemmer

# параметры 

TOP_K = 50
BATCH = 64
SEED = 42

TEXT_CACHE = "item_text_cache.pkl"

COLLAB_TOPK = 50         # сколько ближайших train-запросов учитываем
CLICKS_PER_ITEM = 25     # сколько кликовых запросов подмешиваем в текст объявления
TOP_MICROCAT = 8         # сколько микрокатегорий-кандидатов оставляем
TOP_CATEGORY = 3

W = dict(
    bm25=1.00,
    char=0.55,
    collab=0.85,
    micro=0.35,
    cat=0.15,
    loc=0.20,
    pop=0.06,
    rate=0.03,
)

# нормализация

_stemmer = SnowballStemmer("russian")
_non_word = re.compile(r"[^0-9a-zа-я]+")


@lru_cache(maxsize=2_000_000)
def _stem(token: str) -> str:
    return _stemmer.stem(token)


def norm(s, limit=None):
    if not isinstance(s, str):
        return ""
    if limit:
        s = s[:limit]
    s = _non_word.sub(" ", s.lower().replace("ё", "е")).strip()
    if not s:
        return ""
    return " ".join(_stem(t) for t in s.split())


def norm_series(sr, limit=None):
    return sr.fillna("").astype(str).map(lambda x: norm(x, limit))


def query_text(df):
    q = norm_series(df["search_query"])
    p = norm_series(df.get("search_infm_params_text", pd.Series("", index=df.index)))
    return (q + " " + q + " " + q + " " + p).str.strip()


# BM25

class BM25:
    """Okapi BM25. IDF и нормировка по длине зашиты в матрицу документов,
    запрос подается бинарным вектором термов -> скор = одно sparse-умножение."""

    def __init__(self, k1=1.2, b=0.6, **cv_kwargs):
        self.k1, self.b = k1, b
        self.cv = CountVectorizer(dtype=np.float32,
                                  token_pattern=r"(?u)\b\w+\b", **cv_kwargs)

    def fit(self, docs):
        X = self.cv.fit_transform(docs).tocsr()
        n = X.shape[0]
        dl = np.asarray(X.sum(axis=1)).ravel()
        avgdl = float(dl.mean()) or 1.0
        df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float32)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)

        rows = np.repeat(np.arange(n), np.diff(X.indptr))
        tf = X.data
        X.data = (tf * (self.k1 + 1.0) /
                  (tf + self.k1 * (1 - self.b + self.b * dl[rows] / avgdl))).astype(np.float32)
        X.data *= idf[X.indices]
        self.W = X
        return self

    def scores(self, qtexts):
        Q = self.cv.transform(qtexts).tocsr()
        Q.data = np.ones_like(Q.data)
        return (Q @ self.W.T).toarray()


# вспомогательные функции

def rowmax_norm(a):
    m = a.max(axis=1, keepdims=True)
    np.divide(a, np.where(m > 0, m, 1.0), out=a)
    return a


def rocchio(q_matrix, codes, n_classes):
    """Центроидный классификатор: класс = L2-нормированная сумма
    tf-idf векторов запросов, которые к нему привели."""
    ok = codes >= 0
    idx = np.nonzero(ok)[0]
    M = sparse.csr_matrix(
        (np.ones(idx.size, dtype=np.float32), (codes[idx], idx)),
        shape=(n_classes, q_matrix.shape[0]),
    )
    return normalize(M @ q_matrix).astype(np.float32)


def topk_mask(a, k):
    if a.shape[1] <= k:
        return a
    idx = np.argpartition(-a, k, axis=1)[:, :k]
    out = np.zeros_like(a)
    r = np.arange(a.shape[0])[:, None]
    out[r, idx] = a[r, idx]
    return out


# пайплайн

def run(train, queries, items, verbose=True):
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}")

    item_ids = items["item_id"].astype(str).values
    n_items = len(item_ids)
    pos = {iid: i for i, iid in enumerate(item_ids)}

    train = train.copy()
    train["item_id"] = train["item_id"].astype(str)
    train["_q"] = query_text(train)
    queries = queries.copy()
    queries["_q"] = query_text(queries)
    log("тексты запросов готовы")

    # расширение объявлений кликовыми запросами
    clicks = defaultdict(list)
    for q, iid in zip(train["_q"].values, train["item_id"].values):
        lst = clicks[iid]
        if len(lst) < CLICKS_PER_ITEM and q not in lst:
            lst.append(q)
    click_text = pd.Series([" ".join(clicks.get(i, ())) for i in item_ids],
                           index=items.index)
    covered = sum(1 for i in item_ids if i in clicks)
    log(f"кликовое расширение: {covered}/{n_items} объявлений корпуса "
        f"({100 * covered / n_items:.1f}%) встречались в train")

    # нормализация корпуса не зависит от train - кэшируем между запусками
    if os.path.exists(TEXT_CACHE):
        with open(TEXT_CACHE, "rb") as f:
            title, params, desc = pickle.load(f)
        log("тексты объявлений взяты из кэша")
    else:
        title = norm_series(items["item_title_raw"])
        params = norm_series(items["item_infm_params_text"])
        desc = norm_series(items["item_description_raw"], limit=1200)
        with open(TEXT_CACHE, "wb") as f:
            pickle.dump((title, params, desc), f)
        log("тексты объявлений нормализованы")

    word_text = (title + " " + title + " " + title + " " +
                 params + " " + params + " " + desc + " " +
                 click_text + " " + click_text + " " + click_text)
    char_text = (title + " " + params + " " + click_text).str.slice(0, 300)

    # лексические индексы
    bm25 = BM25(ngram_range=(1, 2), min_df=2, max_features=400_000).fit(word_text)
    log("BM25 построен")

    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3,
                               max_features=250_000, sublinear_tf=True,
                               dtype=np.float32)
    char_items = normalize(char_vec.fit_transform(char_text)).tocsr()
    char_q = normalize(char_vec.transform(queries["_q"])).tocsr()
    log("char-TFIDF построен")

    # пространство запросов (для collab и приоров)
    qvec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    TR = qvec.fit_transform(train["_q"])
    Q = normalize(qvec.transform(queries["_q"])).tocsr()
    TRn = normalize(TR).tocsr()

    uniq_q = train["_q"].drop_duplicates().tolist()
    uq_index = {t: i for i, t in enumerate(uniq_q)}
    UQ = normalize(qvec.transform(uniq_q)).tocsr()

    rows, cols = [], []
    for t, g in train.groupby("_q", sort=False):
        r = uq_index[t]
        for iid in g["item_id"].values:
            p = pos.get(iid)
            if p is not None:
                rows.append(r)
                cols.append(p)
    CL = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(uniq_q), n_items),
    )
    log(f"collaborative-матрица: {CL.shape}, nnz={CL.nnz}")

    # приоры по микрокатегории / категории
    def codes_of(col):
        vals = items[col].fillna("__na__").astype(str)
        c, uniq = pd.factorize(vals)
        return c.astype(np.int32), {v: i for i, v in enumerate(uniq)}, len(uniq)

    mic_code, mic_map, n_mic = codes_of("item_microcat_id")
    cat_code, cat_map, n_cat = codes_of("item_category_id")
    loc_code, loc_map, n_loc = codes_of("item_location_id")

    tr_mic = train["item_microcat_id"].fillna("__na__").astype(str).map(mic_map).fillna(-1).values.astype(np.int32)
    tr_cat = train["item_category_id"].fillna("__na__").astype(str).map(cat_map).fillna(-1).values.astype(np.int32)
    C_MIC = rocchio(TRn, tr_mic, n_mic)
    C_CAT = rocchio(TRn, tr_cat, n_cat)
    log("классификаторы микрокатегории/категории обучены")

    # приор по локации: P(item_location | search_location)
    loc_prior = {}
    tmp = train[["search_location_id", "item_location_id"]].fillna("__na__").astype(str)
    for sl, g in tmp.groupby("search_location_id", sort=False):
        vc = g["item_location_id"].value_counts(normalize=True)
        c = [loc_map[v] for v in vc.index if v in loc_map]
        p = [vc[v] for v in vc.index if v in loc_map]
        if c:
            p = np.asarray(p, dtype=np.float32)
            loc_prior[sl] = (np.asarray(c, dtype=np.int32), p / p.max())

    # популярность и рейтинг
    pop = np.zeros(n_items, dtype=np.float32)
    cnt = train["item_id"].value_counts()
    for iid, c in cnt.items():
        p = pos.get(iid)
        if p is not None:
            pop[p] = np.log1p(c)
    if pop.max() > 0:
        pop /= pop.max()

    rate = (items["item_rating"].fillna(0).astype(np.float32).values / 5.0
            if "item_rating" in items.columns else np.zeros(n_items, dtype=np.float32))

    q_loc = queries["search_location_id"].fillna("__na__").astype(str).values
    log("приоры готовы, старт скоринга")

    # скоринг
    predictions = []
    for start in range(0, len(queries), BATCH):
        end = min(start + BATCH, len(queries))
        sl = slice(start, end)
        b = end - start

        S = rowmax_norm(bm25.scores(queries["_q"].values[sl])) * W["bm25"]
        S += rowmax_norm((char_q[sl] @ char_items.T).toarray()) * W["char"]

        sim = topk_mask((Q[sl] @ UQ.T).toarray(), COLLAB_TOPK)
        S += rowmax_norm((sparse.csr_matrix(sim) @ CL).toarray()) * W["collab"]

        mic = topk_mask(rowmax_norm((Q[sl] @ C_MIC.T).toarray()), TOP_MICROCAT)
        cat = topk_mask(rowmax_norm((Q[sl] @ C_CAT.T).toarray()), TOP_CATEGORY)

        S += W["pop"] * pop + W["rate"] * rate

        for j in range(b):
            s = S[j]
            s += W["micro"] * mic[j][mic_code]
            s += W["cat"] * cat[j][cat_code]

            pr = loc_prior.get(q_loc[start + j])
            if pr is not None:
                v = np.zeros(n_loc, dtype=np.float32)
                v[pr[0]] = pr[1]
                s += W["loc"] * v[loc_code]

            idx = np.argpartition(-s, TOP_K)[:TOP_K]
            idx = idx[np.argsort(-s[idx])]
            predictions.append([item_ids[i] for i in idx])

        if start % (BATCH * 10) == 0:
            log(f"обработано {end}/{len(queries)}")

    log("скоринг завершен")
    return predictions


# ================================ main =================================

def main():
    train = pd.read_parquet("train.parquet")
    items = pd.read_parquet("benchmark_items.parquet")
    queries = pd.read_parquet("benchmark_queries.parquet")
    queries["query_id"] = queries["query_id"].astype(str)

    preds = run(train, queries, items)

    qids = queries["query_id"].tolist()
    corpus = set(items["item_id"].astype(str))

    assert len(qids) == len(set(qids)) == len(preds), "несовпадение query_id"
    for p in preds:
        assert 0 < len(p) <= TOP_K, "некорректное число item_id"
        assert len(p) == len(set(p)), "повтор внутри строки"
        assert all(re.fullmatch(r"[0-9a-f]{16}", i) and i in corpus for i in p), \
            "item_id не из корпуса"

    answer = pd.DataFrame({
        "query_id": qids,
        "answer": [" ".join(p) for p in preds],
    })
    answer.to_csv("answer.csv", index=False, encoding="utf-8")
    print("answer.csv сохранен:", answer.shape)


if __name__ == "__main__":
    main()