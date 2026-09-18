import pandas as pd

answer = pd.read_csv("answer.csv", dtype=str)
queries = pd.read_parquet("benchmark_queries.parquet")
items = pd.read_parquet("benchmark_items.parquet")

queries["query_id"] = queries["query_id"].astype(str)
items["item_id"] = items["item_id"].astype(str)

assert list(answer.columns) == ["query_id", "answer"], "Неверные колонки"
assert len(answer) == len(queries), "Число строк не совпадает"
assert answer["query_id"].is_unique, "query_id повторяются"
assert set(answer["query_id"]) == set(queries["query_id"]), "Не все query_id"

valid_items = set(items["item_id"])

for i, line in enumerate(answer["answer"]):
    ids = line.split()
    assert len(ids) <= 50, f"Строка {i}: больше 50 item_id"
    assert len(ids) == len(set(ids)), f"Строка {i}: повторы item_id"
    for iid in ids:
        assert iid in valid_items, f"Строка {i}: item_id {iid} не найден"

print("answer.csv прошёл все проверки")