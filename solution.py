from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem.snowball import RussianStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


STEMMER = RussianStemmer()


def normalize_text(value: object) -> str:
    text = html.unescape(str(value)).lower().replace("ё", "е")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-zа-я0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stem_text(text: str) -> str:
    return " ".join(
        STEMMER.stem(token) if re.fullmatch(r"[а-я]+", token) else token
        for token in text.split()
    )


def reciprocal_rank_scores(scores: np.ndarray, power: float = 0.25) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked = np.empty_like(scores, dtype=np.float64)
    values = 1.0 / np.power(np.arange(1, scores.shape[1] + 1), power)
    ranked[np.arange(scores.shape[0])[:, None], order] = values
    return ranked


RULES: list[tuple[str, list[int]]] = [
    (r"(сколько|стоим|цен|дорог|сумм|тариф|рассчит).{0,25}достав|достав.{0,25}(сколько|стоим|цен|дорог|сумм|тариф|рассчит)", [1951, 3467, 4234]),
    (r"(как|куда|где).{0,20}(отправ|сдать|передат).{0,25}(заказ|товар|посыл|достав)|как отправить|через какой пункт", [1909, 4234, 4328]),
    (r"(можно|могу|разреш|принима|до скольк|огранич).{0,35}(отправ|достав|посыл)|что можно.{0,20}(отправ|заказ)|\b(вес|габарит|размер).{0,25}(достав|посыл|товар)", [4328, 4396, 4234]),
    (r"(не могу|не получается|невозмож|не доступ|нельзя|нету?|только).{0,35}(отправ|достав|заказ)|достав.{0,25}(не доступ|не актив|нету?|пропал)", [4396, 4308, 4328]),
    (r"(как|хочу|не могу|можно).{0,20}(заказать|оформить).{0,25}(товар|достав|заказ)|как купить.{0,25}достав", [4308, 4234, 4219]),
    (r"упаков|обрешет|хруп|стекл|ювелир|безопасн.{0,20}отправ", [1907, 4328, 4396]),
    (r"крупногабарит|грузов|\bпэк\b|курьер.{0,25}(товар|достав|отправ)|тяжел|коляск.{0,25}(достав|отправ)", [4286, 4328, 4396]),
    (r"продав(ец|ца).{0,35}(не отправ|не высл|не отнес)|товар.{0,25}(не отправил|не выслал)", [1958, 4387, 4219]),
    (r"(когда|куда|как).{0,25}(получ|прид|поступ|забрат|перевест|вывест).{0,25}деньг|деньги.{0,25}(за товар|за продаж|продавц)", [4361, 2943, 4384]),
    (r"отмен|аннулир|остановить.{0,20}(заказ|выдач)", [4387, 1966, 4219]),
    (r"не подош|отказ.{0,20}(товар|получ|заказ)|вернуть.{0,20}(товар|заказ)|возврат.{0,20}(товар|покупк)|брак|поврежд", [4400, 2831, 4387]),
    (r"(когда|через сколько|как долго).{0,30}верн.{0,20}деньг|деньги.{0,25}(не вернул|не приш|вернут)|не вернул.{0,20}деньг", [1966, 2865, 4384, 4219]),
    (r"(верн|возврат).{0,35}достав|достав.{0,35}(верн|возврат)", [2865, 4400, 4219]),
    (r"(не забир|не забрал|отказал).{0,25}(заказ|товар|посыл)|товар.{0,25}(не забир|вернул)", [4403, 4387, 4400]),
    (r"промокод", [2665, 4214]),
    (r"(подключ|добав|сдела|отключ).{0,25}скидк|скидк.{0,25}(продав|мой товар|мое объяв)", [2698, 4214, 4451]),
    (r"скидк", [4214, 4451, 2698]),
    (r"бонус", [4395, 4214, 4424]),
    (r"кошел|баланс", [4384, 2646, 4219]),
    (r"(не могу|не получается|не проход|ошиб).{0,25}(оплат|платеж)|оплат.{0,25}(не проход|ошиб)", [4389, 2646, 4219]),
    (r"(оплат|платеж|деньги спис).{0,40}(объяв|размещ)|объяв.{0,40}(оплат|деньги спис|ждет оплаты)", [4440, 2222, 2886]),
    (r"(размест|подать|вылож|опубликов).{0,30}объяв.{0,30}(ошиб|не могу|не получ)|объяв.{0,30}(ошиб|не публику)", [2222, 4283, 4273]),
    (r"категор", [3254, 2928, 4224]),
    (r"объяв.{0,25}(платн|оплат)|платн.{0,20}(объяв|размещ)", [3128, 4307, 4440]),
    (r"(не виж|не появ|пропал).{0,30}(объяв|объект)|объяв.{0,30}(не виж|не появ|пропал)", [2661, 4283, 4273]),
    (r"автотек", [4232, 4318, 4423]),
    (r"провер(к|ить).{0,25}авто|авто.{0,25}провер", [4423, 4364, 3028]),
    (r"собствен|птс|документ.{0,20}авто", [2908, 3028, 4364]),
    (r"база отдых|путешеств|гост.{0,15}(засел|брон)|засел", [4134, 3993, 4127, 4321]),
    (r"тариф", [2095, 3467, 4276, 4218]),
    (r"одинаков.{0,20}объяв|мульти|объедин.{0,20}объяв", [3862, 4242, 4433]),
    (r"отзыв", [3261, 3147]),
    (r"(где|потер).{0,20}(заказ|посыл)|заказ.{0,20}(потер|не видно)", [3843, 4009, 2511]),
    (r"звонк|устройств.{0,20}звон", [3889, 4133]),
    (r"аукцион", [4178]),
]


PRIORITY_RULES: list[tuple[str, list[int], float]] = [
    (r"подключ.{0,20}тариф", [3580, 3467, 2095], 0.80),
    (r"(попол|внес).{0,30}(кошел|баланс).{0,35}(не|нет)|кошел.{0,30}(попол|не приш)", [4312, 4313, 3077], 0.80),
    (r"(не появ|не виж|исчез|пропал).{0,40}(поиск|поисковик)|поиск.{0,40}(не появ|не виж|исчез|пропал)", [2663, 2253, 2968], 0.70),
    (r"(не публик|не размещ|не вылож).{0,30}объяв", [4008, 2222, 4283], 0.60),
    (r"не ?актив.{0,20}достав|достав.{0,20}не ?актив", [4362, 4396, 3265, 4308], 0.45),
    (r"покупател.{0,30}(не мож|не получ).{0,20}(заказ|оформ)", [4308, 4396, 4234], 0.45),
    (r"(кошел|баланс).{0,45}(не проход|не могу оплат|оплат.*не)", [4384, 4389, 2646], 0.45),
    (r"(заказ|посыл).{0,20}потер|потер.{0,20}(заказ|посыл)", [3843, 2944, 2521], 0.70),
    (r"(?!.*(выплат|кошел|стоим|сумм|комисс|плат)).*((отключ|включ).{0,35}достав|достав.{0,25}(отключ|включ)|личн.{0,15}встреч.{0,65}(способ|достав)|коррект.{0,35}способ.{0,10}достав)", [4362, 4433, 2196], 0.65),
    (r"(за возврат.{0,40}(снима|плат|достав)|возврат.{0,40}(деньг за достав|оплач.{0,10}достав)|обратн.{0,30}достав.{0,20}плат)", [4532, 4400, 2865], 0.80),
    (r"не пришли.{0,20}деньги за товар|деньги за товар.{0,20}не приш", [4361, 2943], 0.45),
    (r"авто.{0,10}выплат", [4361, 4384, 2943], 0.65),
]


def build_answer(
    data_dir: Path,
    output_path: Path,
    use_manual_rules: bool = True,
    manual_rule_boost: float = 0.10,
) -> pd.DataFrame:
    articles = pd.read_feather(data_dir / "articles.f")
    calibration = pd.read_feather(data_dir / "calibration.f")
    test = pd.read_feather(data_dir / "test.f")

    all_queries = pd.concat(
        [calibration["query_text"], test["query_text"]], ignore_index=True
    ).map(normalize_text).tolist()
    stemmed_queries = [stem_text(text) for text in all_queries]

    train_size = len(calibration)
    article_count = len(articles)
    article_ids = articles["article_id"].astype(int).to_numpy()
    article_to_column = {article_id: i for i, article_id in enumerate(article_ids)}

    labels = np.zeros((train_size, article_count), dtype=np.float64)
    for row_number, value in enumerate(calibration["ground_truth"]):
        for article_id in str(value).split():
            labels[row_number, article_to_column[int(article_id)]] = 1.0

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=200_000,
    )
    char_features = char_vectorizer.fit_transform(all_queries)
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, sublinear_tf=True
    )
    stem_features = word_vectorizer.fit_transform(stemmed_queries)

    train_char = char_features[:train_size]
    test_char = char_features[train_size:]
    train_stem = stem_features[:train_size]
    test_stem = stem_features[train_size:]

    train_kernel = 0.5 * (train_char @ train_char.T).toarray()
    train_kernel += 0.5 * (train_stem @ train_stem.T).toarray()
    test_kernel = 0.5 * (test_char @ train_char.T).toarray()
    test_kernel += 0.5 * (test_stem @ train_stem.T).toarray()

    train_kernel = np.square(train_kernel)
    test_kernel = np.square(test_kernel)
    coefficients = np.linalg.solve(
        train_kernel + 0.3 * np.eye(train_size), labels
    )
    supervised_scores = test_kernel @ coefficients

    plain_similarity = 0.5 * (test_char @ train_char.T).toarray()
    plain_similarity += 0.5 * (test_stem @ train_stem.T).toarray()
    neighbour_count = min(10 if use_manual_rules else 5, train_size)
    neighbour_indices = np.argpartition(
        -plain_similarity, neighbour_count - 1, axis=1
    )[:, :neighbour_count]
    neighbour_mask = np.zeros_like(plain_similarity)
    rows = np.arange(len(test))[:, None]
    neighbour_mask[rows, neighbour_indices] = plain_similarity[rows, neighbour_indices]
    neighbour_scores = neighbour_mask @ labels

    article_documents = []
    for row in articles.itertuples(index=False):
        title = normalize_text(row.title)
        body = normalize_text(row.body)
        article_documents.append(((title + " ") * 5 + body).strip())

    direct_corpus = article_documents + all_queries
    direct_char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=220_000,
    )
    direct_char = direct_char_vectorizer.fit_transform(direct_corpus)
    direct_scores = 0.6 * (
        direct_char[article_count + train_size :] @ direct_char[:article_count].T
    ).toarray()

    direct_word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=220_000,
    )
    direct_word = direct_word_vectorizer.fit_transform(direct_corpus)
    direct_scores += 0.4 * (
        direct_word[article_count + train_size :] @ direct_word[:article_count].T
    ).toarray()

    final_scores = 0.95 * (
        0.7 * reciprocal_rank_scores(supervised_scores)
        + 0.3 * reciprocal_rank_scores(direct_scores)
    )
    final_scores += 0.05 * reciprocal_rank_scores(neighbour_scores)

    if use_manual_rules:
        for row_number, query in enumerate(all_queries[train_size:]):
            for pattern, ordered_ids in RULES:
                if re.search(pattern, query):
                    for rank, article_id in enumerate(ordered_ids):
                        column = article_to_column.get(article_id)
                        if column is not None:
                            final_scores[row_number, column] += manual_rule_boost / (1 + 0.5 * rank)
            for pattern, ordered_ids, boost in PRIORITY_RULES:
                if re.search(pattern, query):
                    for rank, article_id in enumerate(ordered_ids):
                        column = article_to_column.get(article_id)
                        if column is not None:
                            final_scores[row_number, column] += boost / (1 + 0.5 * rank)

    top_columns = np.argsort(-final_scores, axis=1, kind="stable")[:, :10]
    answers = [
        " ".join(str(article_ids[column]) for column in row)
        for row in top_columns
    ]
    result = pd.DataFrame(
        {"query_id": test["query_id"].to_numpy(), "answer": answers}
    )
    result.to_csv(output_path, index=False)
    return result


def validate_answer(result: pd.DataFrame, data_dir: Path) -> None:
    articles = pd.read_feather(data_dir / "articles.f")
    test = pd.read_feather(data_dir / "test.f")
    valid_ids = set(articles["article_id"].astype(int))

    assert list(result.columns) == ["query_id", "answer"]
    assert len(result) == len(test)
    assert result["query_id"].tolist() == test["query_id"].tolist()
    assert result["query_id"].is_unique
    for answer in result["answer"]:
        ids = [int(value) for value in str(answer).split()]
        assert len(ids) == 10
        assert len(ids) == len(set(ids))
        assert set(ids) <= valid_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("candidate_data"))
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument(
        "--disable-rules",
        action="store_true",
        help="Use only statistically validated ranking components",
    )
    parser.add_argument("--rule-boost", type=float, default=0.10)
    args = parser.parse_args()

    result = build_answer(
        args.data_dir,
        args.output,
        use_manual_rules=not args.disable_rules,
        manual_rule_boost=args.rule_boost,
    )
    validate_answer(result, args.data_dir)
    print(f"Saved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
