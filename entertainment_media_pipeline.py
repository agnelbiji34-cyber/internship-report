from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_absolute_error, roc_auc_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


MOVIES_DATASET = "rounakbanik/the-movies-dataset"
REVIEWS_DATASET = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def download_from_kaggle(base_dir: Path) -> None:
    import kagglehub

    movies_path = Path(kagglehub.dataset_download(MOVIES_DATASET))
    reviews_path = Path(kagglehub.dataset_download(REVIEWS_DATASET))

    target_movies = base_dir / "movies"
    target_reviews = base_dir / "reviews"
    target_movies.mkdir(parents=True, exist_ok=True)
    target_reviews.mkdir(parents=True, exist_ok=True)

    for source in movies_path.glob("*"):
        if not source.is_file():
            continue
        destination = target_movies / source.name
        if not destination.exists():
            destination.write_bytes(source.read_bytes())

    for source in reviews_path.glob("*"):
        if not source.is_file():
            continue
        destination = target_reviews / source.name
        if not destination.exists():
            destination.write_bytes(source.read_bytes())


def safe_literal_eval(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def extract_names(value, limit: int | None = None) -> list[str]:
    items = safe_literal_eval(value)
    names = [item.get("name", "") for item in items if isinstance(item, dict) and item.get("name")]
    if limit is not None:
        names = names[:limit]
    return names


def extract_director(value) -> str:
    items = safe_literal_eval(value)
    for item in items:
        if isinstance(item, dict) and item.get("job") == "Director":
            return item.get("name", "")
    return ""


def clean_text(value: str) -> str:
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_movies_bundle(data_dir: Path) -> pd.DataFrame:
    movies_dir = data_dir / "movies"
    movies = pd.read_csv(movies_dir / "movies_metadata.csv", low_memory=False)
    credits = pd.read_csv(movies_dir / "credits.csv")
    keywords = pd.read_csv(movies_dir / "keywords.csv")
    ratings = pd.read_csv(movies_dir / "ratings_small.csv")

    movies = movies.rename(columns={"id": "movie_id"})
    movies = movies[movies["movie_id"].astype(str).str.fullmatch(r"\d+")].copy()
    movies["movie_id"] = movies["movie_id"].astype(int)

    credits["id"] = pd.to_numeric(credits["id"], errors="coerce")
    keywords["id"] = pd.to_numeric(keywords["id"], errors="coerce")

    merged = movies.merge(credits, left_on="movie_id", right_on="id", how="left")
    merged = merged.merge(keywords, left_on="movie_id", right_on="id", how="left", suffixes=("", "_keywords"))

    merged["genres_list"] = merged["genres"].apply(extract_names)
    merged["keyword_list"] = merged["keywords"].apply(lambda x: extract_names(x, limit=10))
    merged["cast_list"] = merged["cast"].apply(lambda x: extract_names(x, limit=5))
    merged["director"] = merged["crew"].apply(extract_director)

    numeric_cols = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count"]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")
    ratings_summary = (
        ratings.groupby("movieId")
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )
    merged = merged.merge(ratings_summary, left_on="movie_id", right_on="movieId", how="left")

    merged["release_date"] = pd.to_datetime(merged["release_date"], errors="coerce")
    merged["release_year"] = merged["release_date"].dt.year
    merged["overview"] = merged["overview"].fillna("")
    merged["title"] = merged["title"].fillna(merged["original_title"]).fillna("Unknown")

    merged["tag_text"] = (
        merged["overview"].fillna("")
        + " "
        + merged["genres_list"].apply(lambda x: " ".join(x))
        + " "
        + merged["keyword_list"].apply(lambda x: " ".join(x))
        + " "
        + merged["cast_list"].apply(lambda x: " ".join(x))
        + " "
        + merged["director"].fillna("")
    ).apply(clean_text)

    return merged


def load_reviews(data_dir: Path) -> pd.DataFrame:
    reviews_dir = data_dir / "reviews"
    reviews = pd.read_csv(reviews_dir / "IMDB Dataset.csv")
    reviews["sentiment_label"] = reviews["sentiment"].map({"positive": 1, "negative": 0})
    return reviews


def run_eda(movies: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    genre_counts = (
        movies["genres_list"]
        .explode()
        .dropna()
        .value_counts()
        .head(10)
        .sort_values()
    )

    plt.figure(figsize=(10, 6))
    genre_counts.plot(kind="barh", color="#2563eb")
    plt.title("Top Movie Genres")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "top_genres.png", dpi=200)
    plt.close()

    year_frame = movies.dropna(subset=["release_year"]).copy()
    year_frame["release_year"] = year_frame["release_year"].astype(int)
    releases = year_frame["release_year"].value_counts().sort_index().tail(40)

    plt.figure(figsize=(12, 6))
    releases.plot(color="#dc2626")
    plt.title("Movie Releases by Year")
    plt.xlabel("Year")
    plt.ylabel("Movies")
    plt.tight_layout()
    plt.savefig(output_dir / "release_trend.png", dpi=200)
    plt.close()

    corr_cols = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count", "avg_rating", "rating_count"]
    corr = movies[corr_cols].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    summary = {
        "rows": int(len(movies)),
        "unique_titles": int(movies["title"].nunique()),
        "avg_vote": round(float(movies["vote_average"].mean()), 2),
        "avg_popularity": round(float(movies["popularity"].mean()), 2),
        "median_runtime": round(float(movies["runtime"].median()), 2),
    }
    pd.Series(summary).to_csv(output_dir / "eda_summary.csv", header=["value"])


def build_recommender(movies: pd.DataFrame):
    usable = movies[movies["tag_text"].str.len() > 20].copy()
    usable = usable.drop_duplicates(subset=["title"]).reset_index(drop=True)

    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=12000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(usable["tag_text"])
    similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(usable.index, index=usable["title"].str.lower()).drop_duplicates()
    return usable, similarity, indices


def get_recommendations(title: str, usable: pd.DataFrame, similarity, indices: pd.Series, top_n: int) -> pd.DataFrame:
    idx = indices.get(title.lower())
    if idx is None:
        choices = usable["title"].sort_values().head(20).tolist()
        raise ValueError(f"Title not found. Sample titles: {choices}")

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    picks = [i for i, _ in scores]

    result = usable.loc[picks, ["title", "genres_list", "vote_average", "popularity", "release_year"]].copy()
    result["similarity_score"] = [round(score, 4) for _, score in scores]
    return result


def train_popularity_model(movies: pd.DataFrame) -> tuple[RandomForestRegressor, pd.DataFrame]:
    model_frame = movies.copy()
    model_frame = model_frame.dropna(subset=["popularity"])
    model_frame["genre_count"] = model_frame["genres_list"].apply(len)
    model_frame["is_english"] = (model_frame["original_language"] == "en").astype(int)

    mlb = MultiLabelBinarizer()
    genre_matrix = pd.DataFrame(
        mlb.fit_transform(model_frame["genres_list"]),
        columns=[f"genre_{name}" for name in mlb.classes_],
        index=model_frame.index,
    )

    features = pd.concat(
        [
            model_frame[["budget", "runtime", "vote_average", "vote_count", "revenue", "release_year", "avg_rating", "rating_count", "genre_count", "is_english"]],
            genre_matrix,
        ],
        axis=1,
    )

    y = model_frame["popularity"]
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = pd.DataFrame(
        {
            "metric": ["mae", "baseline_mae"],
            "value": [
                mean_absolute_error(y_test, preds),
                mean_absolute_error(y_test, np.repeat(y_train.mean(), len(y_test))),
            ],
        }
    )

    feature_importance = (
        pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    return model, pd.concat([metrics, feature_importance], axis=0, ignore_index=True)


def train_sentiment_model(reviews: pd.DataFrame) -> tuple[Pipeline, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        reviews["review"],
        reviews["sentiment_label"],
        test_size=0.2,
        random_state=42,
        stratify=reviews["sentiment_label"],
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=25000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1200)),
        ]
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    report = classification_report(y_test, preds, output_dict=True)
    report["roc_auc"] = roc_auc_score(y_test, probs)
    return model, report


def save_sentiment_report(report: dict, output_dir: Path) -> None:
    flat_rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
            flat_rows.append(row)
        else:
            flat_rows.append({"label": label, "value": values})
    pd.DataFrame(flat_rows).to_csv(output_dir / "sentiment_report.csv", index=False)


def save_recommendations(sample_title: str, recommendations: pd.DataFrame, output_dir: Path) -> None:
    recommendations = recommendations.copy()
    recommendations["genres"] = recommendations["genres_list"].apply(lambda x: ", ".join(x))
    recommendations = recommendations.drop(columns=["genres_list"])
    recommendations.to_csv(output_dir / f"recommendations_for_{clean_text(sample_title).replace(' ', '_')}.csv", index=False)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_from_kaggle(data_dir)

    movies = load_movies_bundle(data_dir)
    reviews = load_reviews(data_dir)

    run_eda(movies, output_dir)

    usable_movies, similarity, indices = build_recommender(movies)
    sample_title = args.title.strip() or "The Dark Knight"
    if sample_title.lower() not in indices.index:
        sample_title = usable_movies["title"].iloc[0]
    recommendations = get_recommendations(sample_title, usable_movies, similarity, indices, args.top_n)
    save_recommendations(sample_title, recommendations, output_dir)

    _, importance_table = train_popularity_model(movies)
    importance_table.to_csv(output_dir / "popularity_model_report.csv", index=False)

    _, sentiment_report = train_sentiment_model(reviews)
    save_sentiment_report(sentiment_report, output_dir)

    dashboard = {
        "movies_loaded": int(len(movies)),
        "reviews_loaded": int(len(reviews)),
        "sample_recommendation_seed": sample_title,
        "top_recommendation": recommendations["title"].iloc[0] if not recommendations.empty else None,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as file:
        json.dump(dashboard, file, indent=2)

    print(json.dumps(dashboard, indent=2))


if __name__ == "__main__":
    main()
