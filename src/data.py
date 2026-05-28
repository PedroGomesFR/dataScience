from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Paths


def load_data(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.data_path)


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
