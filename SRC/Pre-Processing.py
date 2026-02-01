"""
Preprocessing utilities
- import preleaned data
- import null value
- Scale data
- Encode Categorical
- Traintest split
    
"""

# Imports
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from SRC.config import DATA_DIR,CLEANED_DATA_PATH,RANDOM_STATE,TEST_SIZE,TARGET_COL
from typing import List,Tuple

# Load pre-cleaned data
def load_data(data_path:Path= CLEANED_DATA_PATH):
    """
    This will load pre-Cleaned data
    Remove white lagging space from column name
    Return clean data
    """
    df = pd.read_csv(data_path)
    
    # Normalize columns 
    df.columns = [str(c).strip() for c in df.columns.to_list()]
    return df

# Extract Categorical and numerical columns
def _extract_cat_cols_num_cols(df:pd.DataFrame):
    cat_cols = df.select_dtypes(include = ["object","category","bool"]).columns.to_list()
    num_cols = df.select_dtypes(include = ["int64","float64","number"]).columns.to_list()
    
    return cat_cols,num_cols

# Build preprocessor
def build_preprocessor(df_or_x:pd.DataFrame):
    # Create a backup
    df = df_or_x.copy()
    
    # If target column is present ,drop the column
    if TARGET_COL in df.columns:
         df = df.drop(columns=TARGET_COL)
    else:
         df = df     
        
    # Extract Numeric and Categorical columns   
    cat_cols,num_cols = _extract_cat_cols_num_cols(df)
    
    # Numeric Pipelines: Impute --> Scale
    num_transformer = Pipeline(steps = [
        ("imputer",SimpleImputer(strategy = "median")),
        ("scaler",StandardScaler())
    ])
    
    # Categorical Pipelines: Impute-> OneHotEncoding
    cat_transformer = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("ohe",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
    ])
    
    # Combine the pipelines  
    transformers = []
    if cat_cols:
        transformers.append(("cat",cat_transformer,cat_cols))
    if num_cols:
        transformers.append(("num",num_transformer,num_cols))
    preprocessor = ColumnTransformer(transformers=transformers,remainder="drop",verbose_feature_names_out=False)    
        
# Train-test split
def split_data(df = pd.DataFrame):
    df = df.copy()
    # If target is missing
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target Column {TARGET_COL} is not found in DataFrame columns:{df.columns.to_list}")
    
    X = df.drop(columns = [TARGET_COL])
    y = df[TARGET_COL]
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)
    return X_train,X_test,y_train,y_test

print("The preprocessing step is complete")


    
