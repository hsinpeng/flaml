# ##### Boston House Prices-Advanced Regression Techniques #####
# Input features in order:
# 1) CRIM: per capita crime rate by town
# 2) ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
# 3) INDUS: proportion of non-retail business acres per town
# 4) CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# 5) NOX: nitric oxides concentration (parts per 10 million) [parts/10M]
# 6) RM: average number of rooms per dwelling
# 7) AGE: proportion of owner-occupied units built prior to 1940
# 8) DIS: weighted distances to five Boston employment centres
# 9) RAD: index of accessibility to radial highways
# 10) TAX: full-value property-tax rate per $10,000 [$/10k]
# 11) PTRATIO: pupil-teacher ratio by town
# 12) B: The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 13) LSTAT: % lower status of the population
# 
# Output variable:
# 1) MEDV: Median value of owner-occupied homes in $1000's [k$]

import sys
import logging
import pandas as pd
from pathlib import Path
import pickle
from flaml import AutoML
from flaml import logger
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def data_analysis(csv):
    try:
        print('--- Data Analysis ---')
        df = pd.read_csv(csv) 
        print(df.head())
    except ValueError as ve:
        return str(ve)   

def data_cleaning(csv, out_csv):
    try:
        print('--- Data Cleaning ---')
        df = pd.read_csv(csv) 
        print(df.head())

        # Values checking
        print(f'Number of missing values: \n{df.isna().sum()}') # Check for missing values
        print(f'Number of duplicates: {df.duplicated().sum()}') # Check for duplicate values

        df.to_csv(out_csv, index=False) 
    except ValueError as ve:
        return str(ve) 
     
def train(train_X, train_y, model): 
    try:
        print('--- Training ---')
        automl = AutoML()
        # Specify automl goal and constraint
        automl_settings = {
            "time_budget": 300,  # in seconds
            "metric": "mse",
            "task": "regression",
            "estimator_list":['lgbm', 'rf', 'extra_tree', 'kneighbor'],
            "log_file_name": "churn.log",
        }
        # Train with labeled input data
        automl.fit(X_train=train_X, y_train=train_y, **automl_settings)
        # Save model
        with open(model, "wb") as f:
            pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
        # Predict
        print(automl.predict(train_X))
        # Print the best model
        print(automl.model.estimator)
    except ValueError as ve:
        return str(ve)

def test(model, test_X, test_y, result):
    try:
        print('--- Testing ---')
        with open(model, "rb") as f:
            automl = pickle.load(f)
        pred_y = automl.predict(test_X)

        mse = mean_squared_error(test_y, pred_y)
        r2 = r2_score(test_y, pred_y)

        print(f"Mean Squared Error: {mse}")
        print(f"R2 Score: {r2}")

    except ValueError as ve:
        return str(ve)

def main():
    run_option = 1
    try:
        print("Hello FLAML!")
        logger.setLevel(logging.ERROR)
        csv_file = './datasets/boston.csv'
        clean_csv_file = './output/boston_cleaned.csv'
        out_model = './output/boston.pkl'
        out_result = './output/boston.txt'

        match run_option:
            case 0:
                data_analysis(csv_file)
            case 1:
                data_cleaning(csv_file, clean_csv_file)
                df = pd.read_csv(clean_csv_file)
                X = df.drop(['MEDV'], axis=1)
                y = df['MEDV']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                train(X_train, y_train, out_model)
                test(out_model, X_test, y_test, out_result)
            case _:
                print(f'run_option={run_option}')
                print(f'Error: Wrong run_option!')

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())