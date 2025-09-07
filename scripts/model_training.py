from utils import (load_csv, split_xy ,
           train_on_train_with_threshold, check_missing_and_duplicates, apply_smote)
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pandas as pd
import joblib

import os
import warnings
warnings.filterwarnings("ignore")

# Load Data
df = load_csv (r'./data/split/trainval.csv')
# Check for missing values and duplicates
_ , df = check_missing_and_duplicates(df, drop_duplicates=True)
# Split features and target
X_train, y_train = split_xy(df, target_col='Class')

# apply SMOTE to handle class imbalance
X_train, y_train = apply_smote(X_train, y_train)
print(f"After SMOTE, counts of label '1': {sum(y_train==1)}")

# Build Pipeline

pipe = Pipeline([
    ("robust", RobustScaler()),
    ("minmax", MinMaxScaler()),
    ("model", XGBClassifier(scale_pos_weight=10,  
                            random_state=42,eval_metric='logloss',
                            n_estimators=300 , max_depth =6 , learning_rate = 0.2,))
])

# Train model with threshold tuning and save the model

 train_on_train_with_threshold(X_train, y_train, pipe, 
                             save_path=r'./models/xgb_model.pkl', threshold=0.872553)