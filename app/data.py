"""Helpers pour le chargement et la préparation des données.

Ce module contient les fonctions utilisées par l'application Streamlit
pour charger, imputer et préparer les données du jeu Ames Housing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(file) -> pd.DataFrame:
    """Charge un CSV et ajoute quelques features dérivées."""
    df = pd.read_csv(file)
    if "YearBuilt" in df.columns:
        df["AgeLogement"] = 2025 - df["YearBuilt"]
    if all(c in df.columns for c in ["GrLivArea", "TotalBsmtSF"]):
        df["SurfaceTotale"] = df["GrLivArea"] + df["TotalBsmtSF"].fillna(0)
    if all(c in df.columns for c in ["FullBath", "HalfBath"]):
        df["NbSallesDeBain"] = df["FullBath"] + 0.5 * df["HalfBath"]
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les valeurs manquantes selon une stratégie raisonnable."""
    df = df.copy()

    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType",
    ]
    zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
    ]

    for c in none_cols:
        if c in df.columns:
            df[c] = df[c].fillna("None")

    for c in zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    for c in df.columns:
        if df[c].isnull().any():
            if df[c].dtype == "object":
                df[c] = df[c].fillna(df[c].mode()[0])
            else:
                df[c] = df[c].fillna(df[c].median())

    return df


def prepare_data(df_raw: pd.DataFrame, test_sz: float, log_target: bool, remove_outliers: bool = False):
    """Prépare les données pour l'entraînement : imputation, feature engineering, split."""
    df = impute(df_raw.copy())
    if "SalePrice" not in df.columns:
        return None, None, None, None, None, {}, 0

    if remove_outliers and all(c in df.columns for c in ["GrLivArea", "SalePrice"]):
        outlier_mask = ~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300_000))
        n_removed = (~outlier_mask).sum()
        df = df[outlier_mask].reset_index(drop=True)
    else:
        n_removed = 0

    y = np.log1p(df["SalePrice"]) if log_target else df["SalePrice"]
    X = df.drop(columns=["SalePrice", "Id"], errors="ignore")

    # Re-feature engineering
    if "YearBuilt" in X.columns:
        X["AgeLogement"] = 2025 - X["YearBuilt"]
    if all(c in X.columns for c in ["GrLivArea", "TotalBsmtSF"]):
        X["SurfaceTotale"] = X["GrLivArea"] + X["TotalBsmtSF"].fillna(0)
    if all(c in X.columns for c in ["FullBath", "HalfBath"]):
        X["NbSallesDeBain"] = X["FullBath"] + 0.5 * X["HalfBath"]

    # Encode catégoriels
    les = {}
    for c in X.select_dtypes("object").columns:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        les[c] = le

    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_sz / 100, random_state=42)
    return X_tr, X_te, y_tr, y_te, X.columns.tolist(), les, n_removed
