"""Module d'entraînement et d'évaluation des modèles."""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_model_defs(n_estimators: int, max_depth: int, lr_gb: float) -> dict:
    md_val = None if max_depth == 0 else max_depth
    return {
        "Régression Linéaire": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
        "Random Forest": RandomForestRegressor(
            n_estimators=n_estimators, max_depth=md_val,
            min_samples_split=5, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=lr_gb,
            max_depth=4, subsample=0.8, random_state=42
        ),
    }


def train_and_evaluate(X_train, y_train, X_test, y_test, model_defs: dict, target_log: bool):
    """Entraîne les modèles et calcule les métriques pour chaque modèle."""
    results = {}

    for name, model in model_defs.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = (mean_absolute_error(np.expm1(y_test), np.expm1(preds))
               if target_log else mean_absolute_error(y_test, preds))
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        results[name] = {
            "model": model,
            "preds": preds,
            "R²": round(r2, 4),
            "MAE": round(mae, 0),
            "RMSLE": round(rmse, 4),
            "CV_mean": round(cv.mean(), 4),
            "CV_std": round(cv.std(), 4),
        }

    return results
