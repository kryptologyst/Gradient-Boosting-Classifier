import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from loguru import logger
from typing import Tuple

DATASETS = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}

def load_classification_data(dataset_name="wine", test_size=0.2, random_state=42):
    loader = DATASETS.get(dataset_name, load_wine)
    data = loader()
    X, y = data.data, data.target
    fn = list(data.feature_names)
    cn = list(data.target_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    logger.info(f"{dataset_name}: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test, fn, cn
