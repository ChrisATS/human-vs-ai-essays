from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: 'text' (string), 'generated' ("0" or "1")
    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'generated' columns")
    # Normalize label to int 0/1
    df["generated"] = df["generated"].astype(str).str.strip().astype(int)
    return df[["text", "generated"]].dropna()


def train_test_split_stratified(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    X = df["text"].astype(str).tolist()
    y = df["generated"].astype(int).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

