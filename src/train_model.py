import argparse
import os
import joblib
import pandas as pd
import numpy as np
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class NormalizeText(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.Series(X).astype(str).str.strip().str.lower().tolist()


class TitleFeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        feats = []
        for text in X:
            tokens = text.split()
            num_tokens = len(tokens)
            num_chars = len(text)
            has_number = int(bool(re.search(r"\d", text)))
            longest_word_len = max((len(t) for t in tokens), default=0)
            feats.append([num_tokens, num_chars, has_number, longest_word_len])
        return np.array(feats, dtype=float)


class TitleFeaturesPipeline(Pipeline):
    def __init__(self):
        super().__init__([
            ("title_feats", TitleFeaturesExtractor()),
            ("scaler", StandardScaler())
        ])


def build_pipeline():
    text_branch = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        ))
    ])

    numeric_branch = TitleFeaturesPipeline()

    features = FeatureUnion([
        ("text", text_branch),
        ("num", numeric_branch)
    ])

    model = Pipeline([
        ("normalize", NormalizeText()),
        ("features", features),
        ("clf", LinearSVC())
    ])

    return model


def clean_categories(raw_series: pd.Series) -> pd.Series:
    mapping = {
        "fridge": "fridges",
        "cpu": "cpus",
        "mobile phone": "mobile phones",
    }
    return (
        raw_series
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(mapping)
    )


def load_and_prepare(data_path: str):
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    df["clean_category"] = clean_categories(df["Category Label"])
    df = df.dropna(subset=["Product Title", "clean_category"])

    X = df["Product Title"]
    y = df["clean_category"]
    return X, y


def main(args):
    X, y = load_and_prepare(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 4))
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (stvarna klasa po redovima, predikcija po kolonama):")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(pipeline, args.model_path)
    print()
    print(f"Model je sačuvan u: {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treniranje modela za predikciju kategorije proizvoda na osnovu naslova."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "IMLP4_TASK_03-products.csv"),
        help="Putanja do CSV fajla sa proizvodima."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("models", "product_category_model.pkl"),
        help="Gde da sačuvamo istrenirani model (.pkl)."
    )

    args = parser.parse_args()
    main(args)
