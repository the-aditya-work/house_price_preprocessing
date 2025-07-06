# house_price_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Fill missing categorical features with 'None'
cat_cols_to_fill = ['Alley', 'BsmtQual', 'BsmtCond', 'FireplaceQu',
                    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                    'PoolQC', 'Fence', 'MiscFeature']

for col in cat_cols_to_fill:
    train_df[col] = train_df[col].fillna("None")
    test_df[col] = test_df[col].fillna("None")

# Fill missing numerical features with median
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(test_df[col].median())

# Fill any remaining missing values in test set
test_df = test_df.fillna(test_df.median(numeric_only=True))

# Label Encoding for all categorical columns
cat_cols = train_df.select_dtypes(include=['object']).columns

# Ensure all categorical columns have no missing values before encoding
for col in cat_cols:
    train_df[col] = train_df[col].fillna("None")
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna("None")

# Use a separate LabelEncoder for each column, fit on combined data
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    if col in test_df.columns:
        test_df[col] = le.transform(test_df[col].astype(str))

# Feature Engineering
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']

train_df['HouseAge'] = train_df['YrSold'] - train_df['YearBuilt']
test_df['HouseAge'] = test_df['YrSold'] - test_df['YearBuilt']

train_df['Remodeled'] = (train_df['YearBuilt'] != train_df['YearRemodAdd']).astype(int)
test_df['Remodeled'] = (test_df['YearBuilt'] != test_df['YearRemodAdd']).astype(int)

# Feature Scaling
scaler = StandardScaler()
scale_cols = ['TotalSF', 'GrLivArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'HouseAge']

train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
test_df[scale_cols] = scaler.transform(test_df[scale_cols])

# Print confirmation
print("✅ Data Preprocessing and Feature Engineering Completed")