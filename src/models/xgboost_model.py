import numpy as np
import xgboost as xgb
from typing import Any, Optional

from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 7,
        learning_rate: float = 0.01,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        objective: str = "reg:squarederror",
        num_class: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "multi:softprob" if num_class else objective,
            "num_class": num_class,
            "random_state": 42,
        }
        self.num_class = num_class
        self.model: xgb.XGBModel = None

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray = None, y_val: np.ndarray = None,
    ) -> Any:
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], -1)

        params = dict(self.params)
        if self.early_stopping_rounds:
            params["early_stopping_rounds"] = self.early_stopping_rounds

        if self.num_class:
            self.model = xgb.XGBClassifier(**params)
        else:
            self.model = xgb.XGBRegressor(**params)

        fit_kwargs = {"verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str) -> "XGBoostModel":
        cls = xgb.XGBClassifier if self.num_class else xgb.XGBRegressor
        self.model = cls()
        self.model.load_model(path)
        return self

    @property
    def name(self) -> str:
        return "xgboost"
