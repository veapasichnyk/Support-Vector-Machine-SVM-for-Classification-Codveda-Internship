"""
Preprocessing utilities for the churn-bigml dataset.

This module provides:
- A stratified 3-way splitter (train/validation/test)
- Helpers to identify column types and remove unwanted columns
- A configurable ColumnTransformer preprocessor (impute + optional scale + one-hot)
- Two preprocessing entry points:
    * preprocess_churn_data: 2-way split (train/validation)
    * preprocess_churn_data_split: 3-way split (train/validation/test)
- A function to preprocess new/unseen data with previously fitted transformers

The target column is assumed to be "Churn" (bool, object "True"/"False", or numeric 0/1).
"""

from typing import Tuple, List, Optional, Any
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse


def split_train_val_test(
    df: pd.DataFrame,
    target_col: str,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified 3-way split into train, validation, and test subsets.
    """
    train_temp, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_temp,
        test_size=val_size / (1 - test_size),
        stratify=train_temp[target_col],
        random_state=random_state
    )
    return train_df, val_df, test_df


def split_features_targets(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    """Split the DataFrame into features (X) and target (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def remove_unwanted_columns(X: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
    """Remove specified columns from the DataFrame (ignore if not present)."""
    return X.drop(columns=columns_to_remove, errors="ignore")


def get_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical columns in the DataFrame."""
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], scaler_numeric: bool) -> ColumnTransformer:
    """Build ColumnTransformer for numeric (impute, optional scale) and categorical (impute, one-hot)."""
    numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaler_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])


def _ensure_binary_target(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Ensure the target column is encoded as 0/1 integers."""
    df = df.copy()
    if df[target_col].dtype == "bool":
        df[target_col] = df[target_col].astype(int)
    elif df[target_col].dtype == "object":
        df[target_col] = df[target_col].map({"True": 1, "False": 0, "Yes": 1, "No": 0})
    return df


def preprocess_churn_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = False
) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, List[str], Optional[StandardScaler], OneHotEncoder, List[str]]:
    """
    Two-way preprocessing (train/validation) for churn-bigml data.
    """
    target_col = "Churn"
    columns_to_remove: List[str] = []

    raw_df = _ensure_binary_target(raw_df, target_col=target_col)

    X, y = split_features_targets(raw_df, target_col)
    X = remove_unwanted_columns(X, columns_to_remove)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    numeric_cols, categorical_cols = get_column_types(X_train)
    input_cols = numeric_cols + categorical_cols

    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scaler_numeric)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    scaler = preprocessor.named_transformers_["num"].named_steps.get("scaler") if scaler_numeric else None
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_features = encoder.get_feature_names_out(categorical_cols).tolist()
    final_feature_names = numeric_cols + cat_features

    return X_train_processed, y_train, X_val_processed, y_val, input_cols, scaler, encoder, final_feature_names


def preprocess_churn_data_split(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = False,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, np.ndarray, pd.Series, List[str]]:
    """
    Three-way preprocessing (train/validation/test) for churn-bigml data.
    """
    target_col = "Churn"
    raw_df = _ensure_binary_target(raw_df, target_col=target_col)

    train_df, val_df, test_df = split_train_val_test(
        raw_df, target_col=target_col,
        val_size=val_size, test_size=test_size, random_state=random_state
    )

    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_val, y_val = val_df.drop(columns=[target_col]), val_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    numeric_cols, categorical_cols = get_column_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scaler_numeric)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_features = encoder.get_feature_names_out(categorical_cols).tolist()
    feature_names = numeric_cols + cat_features

    return X_train_processed, y_train, X_val_processed, y_val, X_test_processed, y_test, feature_names


def preprocess_new_churn_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder
) -> np.ndarray:
    """Preprocess new/unseen churn-like data using previously fitted scaler/encoder."""
    df = new_df.copy()
    df = remove_unwanted_columns(df, [])
    df = df[input_cols]

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    if scaler:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    if categorical_cols:
        encoded_cats = encoder.transform(df[categorical_cols])
        if issparse(encoded_cats):
            encoded_cats = encoded_cats.toarray()
        encoded_cat_df = pd.DataFrame(encoded_cats, index=df.index)
        df = df.drop(columns=categorical_cols)
        df_final = pd.concat([df.reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)
    else:
        df_final = df.reset_index(drop=True)

    return df_final.values