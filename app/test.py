"""Module de prédiction et d'évaluation finale."""

import numpy as np
import pandas as pd


def predict_from_input(model, row: pd.DataFrame, feat_cols: list, label_encoders: dict, target_log: bool):
    """Encode et aligne une ligne d'entrée, puis prédit le prix."""
    row = row.copy()

    # Encoder les catégorielles avec les label encoders du modèle
    for c in row.select_dtypes("object").columns:
        if c in label_encoders:
            le = label_encoders[c]
            val = str(row[c].iloc[0])
            row[c] = le.transform([val])[0] if val in le.classes_ else 0
        else:
            row[c] = 0

    # Aligner les colonnes exactement comme celles utilisées par le modèle
    missing_cols = [c for c in feat_cols if c not in row.columns]
    for mc in missing_cols:
        row[mc] = 0

    row = row[feat_cols]

    pred_raw = model.predict(row)[0]
    return np.expm1(pred_raw) if target_log else pred_raw
