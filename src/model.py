import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from loguru import logger
from typing import Optional
import joblib


class GradientBoostingModel:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.feature_names_: list = []
        self.class_names_: list = []

    def fit(self, X, y, feature_names=None, class_names=None):
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.class_names_ = class_names or [f"c{c}" for c in np.unique(y)]
        self.model.fit(X, y)
        logger.info(f"GB fitted: {self.n_estimators} trees, lr={self.learning_rate}")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "classification_report": classification_report(
                y, y_pred, target_names=self.class_names_, output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

    def get_feature_importance(self):
        return dict(zip(self.feature_names_, self.model.feature_importances_))

    def grid_search(self, X, y, cv=5):
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        }
        grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_grid, cv=cv, scoring="accuracy", n_jobs=-1,
        )
        grid.fit(X, y)
        logger.info(f"Best GB: {grid.best_params_} → {grid.best_score_:.3f}")
        return {"best_params": grid.best_params_, "best_score": float(grid.best_score_)}

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
