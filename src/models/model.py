from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV


def build_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),  # unigrams + bigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    clf = LinearSVC(C=1.0)
    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])
    return pipe


def train_pipeline(model: Pipeline, X_train, y_train) -> Pipeline:
    model.fit(X_train, y_train)
    return model


def evaluate_pipeline(model: Pipeline, X_test, y_test) -> Dict[str, float]:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def tune_pipeline(
    X, y,
    param_grid: dict | None = None,
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 1,
):
    """Grid search hyperparameters for the TF-IDF + LinearSVC pipeline.

    Returns (best_estimator, best_params, best_score).
    """
    base = build_pipeline()
    if param_grid is None:
        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "clf__C": [0.5, 1.0, 2.0],
        }
    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, gs.best_score_
