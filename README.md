# house_price_preprocessing

This project provides a robust data preprocessing pipeline for house price prediction datasets, such as those from Kaggle's House Prices: Advanced Regression Techniques competition. The script handles missing values, encodes categorical variables, engineers new features, and scales numerical features, preparing the data for machine learning models.

## Features

- Handles missing values for both categorical and numerical columns
- Label encodes categorical variables for model compatibility
- Engineers new features such as total square footage and house age
- Identifies and flags remodeled houses
- Scales key numerical features using standardization

## Usage

1. Place your `train.csv` and `test.csv` files in the project directory.
2. Run the preprocessing script:

   ```sh
   python house_price_preprocessing.py

## Requirements

Python 3.x
pandas
numpy
scikit-learn

## Install dependencies with:

```sh
pip install pandas numpy scikit-learn
