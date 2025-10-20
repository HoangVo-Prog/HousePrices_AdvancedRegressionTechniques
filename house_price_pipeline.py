# house_price_pipeline.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, FunctionTransformer
from sklearn.impute import SimpleImputer

# ===== 1) Canonical ordinal map for Ames-like data =====
ORDINAL_MAP_CANONICAL: Dict[str, List[str]] = {
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageFinish": ["NA", "Unf", "RFn", "Fin"],
    "GarageQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["NA", "Fa", "TA", "Gd", "Ex"],
    "Fence": ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "PavedDrive": ["N", "P", "Y"],
    "Street": ["Grvl", "Pave"],
    "Alley": ["NA", "Grvl", "Pave"],
    "CentralAir": ["N", "Y"]
}

# ===== 2) Small utilities =====
class OrdinalMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping: Dict[str, List[str]]):
        self.mapping = mapping
        self.maps_: Dict[str, Dict[str, int]] = {}

    def fit(self, X, y=None):
        self.maps_ = {}
        for col, order in self.mapping.items():
            if col in X.columns:
                self.maps_[col] = {v: i for i, v in enumerate(order)}
        return self

    def transform(self, X):
        X = X.copy()
        for col, m in self.maps_.items():
            X[col] = X[col].map(m)
        return X

class MissingnessIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, cols: Optional[List[str]] = None, auto_numeric: bool = True):
        self.cols = cols
        self.auto_numeric = auto_numeric
        self.cols_: List[str] = []

    def fit(self, X, y=None):
        if self.cols is not None:
            self.cols_ = [c for c in self.cols if c in X.columns]
        elif self.auto_numeric:
            num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            self.cols_ = [c for c in num_cols if X[c].isna().any()]
        else:
            self.cols_ = []
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols_:
            X[f"{c}_was_missing"] = X[c].isna().astype(int)
        return X

class RarePooler(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str], min_count: int = 20):
        self.cols = cols
        self.min_count = min_count
        self.keep_levels_: Dict[str, set] = {}

    def fit(self, X, y=None):
        self.keep_levels_ = {}
        for c in self.cols:
            if c in X.columns:
                vc = X[c].value_counts(dropna=False)
                self.keep_levels_[c] = set(vc[vc >= self.min_count].index.astype(str))
        return self

    def transform(self, X):
        X = X.copy()
        for c, keep in self.keep_levels_.items():
            if c in X.columns:
                X[c] = X[c].astype(str)
                X[c] = np.where(X[c].isin(keep), X[c], "Other")
        return X

class FiniteCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mask = ~np.isfinite(X)
        if mask.any():
            X[mask] = np.nan
        return X

class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.keep_idx_ = np.where(~np.all(np.isnan(X), axis=0))[0]
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_idx_]

class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, alpha: float = 20.0):
        self.cols = cols
        self.alpha = float(alpha)
        self.global_mean_ = None
        self.maps_ = {}

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TargetEncoderTransformer requires y in fit.")
        y = pd.Series(y).astype(float)
        self.global_mean_ = float(y.mean())
        self.maps_ = {}
        if self.cols is None:
            return self
        Xc = X.copy()
        for c in self.cols:
            if c not in Xc.columns:
                continue
            g = Xc[c].astype(str).groupby(Xc[c].astype(str))
            stats = g.size().to_frame("cnt")
            stats["sumy"] = y.groupby(Xc[c].astype(str)).sum()
            stats["te"] = (stats["sumy"] + self.alpha * self.global_mean_) / (stats["cnt"] + self.alpha)
            self.maps_[c] = stats["te"].to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for c, m in self.maps_.items():
            if c in X.columns:
                X[f"TE_{c}"] = X[c].astype(str).map(m).fillna(self.global_mean_).astype(float)
        return X

# ===== 3) Domain features for Ames =====
def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["TotalBsmtSF","1stFlrSF","2ndFlrSF"]:
        if c not in df.columns: df[c] = 0
    df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)

    for c in ["FullBath","HalfBath","BsmtFullBath","BsmtHalfBath"]:
        if c not in df.columns: df[c] = 0
    df["TotalBath"] = df["FullBath"].fillna(0) + 0.5*df["HalfBath"].fillna(0) + df["BsmtFullBath"].fillna(0) + 0.5*df["BsmtHalfBath"].fillna(0)

    for c in ["YrSold","YearBuilt","YearRemodAdd","GarageYrBlt"]:
        if c not in df.columns: df[c] = np.nan
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).astype(float)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).astype(float)
    df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).astype(float)

    df["IsRemodeled"] = (df.get("YearRemodAdd", df["YearBuilt"]) != df["YearBuilt"]).astype(int)
    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)

    for c in ["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","WoodDeckSF"]:
        if c not in df.columns: df[c] = 0
    df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]

    df["BathPerBedroom"] = df.get("TotalBath", 0) / np.maximum(df.get("BedroomAbvGr", 1), 1)
    df["RoomsPerArea"] = df.get("TotRmsAbvGrd", 0) / np.maximum(df.get("GrLivArea", 1), 1)
    df["LotAreaRatio"] = df.get("LotArea", 0) / np.maximum(df.get("GrLivArea", 1), 1)

    if "MoSold" in df.columns:
        df["MoSold_sin"] = np.sin(2*np.pi*(df["MoSold"].astype(float)/12.0))
        df["MoSold_cos"] = np.cos(2*np.pi*(df["MoSold"].astype(float)/12.0))

    if ("Neighborhood" in df.columns) and ("BldgType" in df.columns):
        df["Neighborhood_BldgType"] = df["Neighborhood"].astype(str) + "|" + df["BldgType"].astype(str)

    df["Ln_TotalSF"] = np.log1p(df.get("TotalSF", 0).astype(float))

    if "OverallQual" in df.columns:
        df["IQ_OQ_GrLiv"] = df["OverallQual"].astype(float) * df.get("GrLivArea", 0).astype(float)
        df["IQ_OQ_TotalSF"] = df["OverallQual"].astype(float) * df.get("TotalSF", 0).astype(float)

    # Optional: mild winsorization for heavy tails (safe for linear models)
    if "LotArea" in df.columns:
        q_hi = df["LotArea"].quantile(0.99)
        df["LotArea_clip"] = np.minimum(df["LotArea"], q_hi)

    return df

# ===== 4) Feature list builder =====
def build_feature_lists(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    for df in (df_train, df_test):
        if "MSSubClass" in df.columns:
            df["MSSubClass"] = df["MSSubClass"].astype(str)

    all_cols = df_train.drop(columns=["SalePrice"], errors="ignore").columns

    ord_cols = [c for c in ORDINAL_MAP_CANONICAL.keys() if c in all_cols]
    cat_cols = [c for c in all_cols if (df_train[c].dtype == "object") and (c not in ord_cols)]
    num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df_train[c]) and c not in ord_cols]

    num_abs_candidates = []
    for c in num_cols:
        if df_train[c].isna().any() or df_test[c].isna().any():
            num_abs_candidates.append(c)

    num_cont = [c for c in num_cols if c not in num_abs_candidates]
    return cat_cols, ord_cols, num_cont, num_abs_candidates

# ===== 5) ColumnTransformer-based preprocessor =====
def make_preprocessor(cat_cols: List[str], ord_cols: List[str], num_cont: List[str], num_absence: List[str]):
    TE_DEFAULT = ["Neighborhood","MSZoning","Exterior1st","Exterior2nd","SaleCondition","BldgType","Neighborhood_BldgType"]

    pre_steps = [
        ("ordinal_map", OrdinalMapper(ORDINAL_MAP_CANONICAL)),
        ("missing_flags", MissingnessIndicator(cols=None, auto_numeric=True)),
        ("rare_pool", RarePooler(cat_cols, min_count=15)),
        ("te", TargetEncoderTransformer(cols=[c for c in TE_DEFAULT if c in cat_cols], alpha=30.0)),
    ]
    pre = Pipeline(steps=pre_steps)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    ord_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    num_cont_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("qntl", QuantileTransformer(output_distribution="normal", n_quantiles=200, subsample=200000, copy=True)),
    ])

    num_abs_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("cats", cat_pipe, cat_cols),
            ("ords", ord_pipe, ord_cols),
            ("num_cont", num_cont_pipe, num_cont),
            ("num_abs", num_abs_pipe, num_absence),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return Pipeline([("prep", pre), ("ct", ct), ("finite", FiniteCleaner()), ("dropnan", DropAllNaNColumns())])

# ===== 6) Factory: full feature space with domain features first =====
def make_feature_space(df_train: pd.DataFrame, df_test: Optional[pd.DataFrame] = None) -> Pipeline:
    if df_test is None:
        df_test = df_train
    df_train_aug = add_domain_features(df_train.copy())
    df_test_aug = add_domain_features(df_test.copy())
    cat_cols, ord_cols, num_cont, num_abs = build_feature_lists(df_train_aug, df_test_aug)

    return Pipeline([
        ("add_domain", FunctionTransformer(add_domain_features)),
        ("preproc", make_preprocessor(cat_cols, ord_cols, num_cont, num_abs)),
    ])
