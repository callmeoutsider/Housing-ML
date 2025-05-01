"""
Pre-processing utilities for the Housing-ML project.
Creates a scikit-learn ColumnTransformer that:
- imputes missing values
- scales numeric features
- one-hot-encodes the categorical column 'ocean_proximity'
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import List

NUMERIC_COLS: List[str] = [
    "MedInc",            # median income
    "HouseAge",          # housing median age
    "AveRooms",          # average rooms
    "AveBedrms",         # average bedrooms
    "Population",
    "AveOccup"           # average household size
]

CATEGORICAL_COLS = ["ocean_proximity"]


def get_preprocessor() -> ColumnTransformer:
    """Return a fitted-later ColumnTransformer for the pipeline."""
    numeric_pipeline = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]

    categorical_pipeline = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )
    return preprocessor
