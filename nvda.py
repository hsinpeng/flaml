# ##### NVIDIA Stock Historical Dataset #####
# Date: Represents the date of each trading day.
# Open: Shows the opening price of NVIDIA's stock on each trading day.
# High: Indicates the highest price reached by NVIDIA's stock during each trading day.
# Low: Indicates the lowest price of NVIDIA's stock observed on each trading day.
# Close: Represents the closing price of NVIDIA's stock for each trading day.
# Adj Close: Represents the adjusted closing price of NVIDIA's stock, accounting for corporate actions.
# Volume: Represents the trading volume of NVIDIA's stock for each trading day, indicating the number of shares traded.

import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import pickle
from flaml import AutoML
from flaml import logger
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def data_analysis(csv):
    try:
        print('--- Data Analysis ---')
        df = pd.read_csv(csv)
        print(df.columns) # Check column names
        col_list = df.columns.to_list()
        print(col_list)
        print(df.info())
        print(df.head())
        print(df.describe().T)
        #df.describe().T.plot(kind='bar')
        #plt.show()

        # Convert 'Date' column to datetime format if necessary
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        # Correlation matrix plot
        corr_matrix = df.corr()
        plt.figure(figsize=(10,6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

        # Line plot
        #fig, ax = plt.subplots(figsize=(20, 8))
        #ax.plot(df['Date'], df['Close'], color='green')
        #ax.xaxis.set_major_locator(plt.MaxNLocator(15))
        #ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
        #ax.set_xlabel('Date', fontsize=14)
        #ax.set_ylabel('Price in USD', fontsize=14)
        #plt.title('NVIDIA Stock Historical Prices', fontsize=18)
        #plt.grid()
        #plt.show()

        # Bar plot
        #fig2, ax = plt.subplots(figsize=(20, 8))
        #ax.bar(df['Date'], df['Close'], color='green')
        #ax.xaxis.set_major_locator(plt.MaxNLocator(15))
        #ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
        #ax.set_xlabel('Date', fontsize=14)
        #ax.set_ylabel('Price in USD', fontsize=14)
        #plt.title('NVIDIA Stock Historical Prices', fontsize=18)
        #plt.grid()
        #plt.show()

        # Histogram plot
        df.hist(bins = 20, figsize = (20,20), color = 'g')
        plt.show()
    except ValueError as ve:
        return str(ve)   
    
def data_cleaning(csv, out_csv):
    try:
        print('--- Data Cleaning ---')
        df = pd.read_csv(csv)

        # Convert 'Date' column to datetime format if necessary
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            #print(df.head())
        
        # Values checking
        print(f'Number of missing values: \n{df.isna().sum()}') # Check for missing values
        print(f'Number of duplicates: {df.duplicated().sum()}') # Check for duplicate values

        # Simple Moving Average (SMA): Calculated by averaging the stock prices over a specified number of periods.
        df['50_MA'] = df['Adj Close'].rolling(window=50).mean() # Calculate the 50-day moving average
        df['200_MA'] = df['Adj Close'].rolling(window=200).mean() # Calculate the 200-day moving average
        
        # Exponential Moving Average (EMA): Calculated by applying more weight to recent stock prices, the Exponential Moving Average (EMA) is more responsive to new information and recent price movements, making it a valuable tool for identifying short-term trends and market signals.
        df['50_EMA'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
        df['200_EMA'] = df['Adj Close'].ewm(span=200, adjust=False).mean()

        # Daily Returns: Daily returns are calculated as the percentage change in the adjusted closing price from one day to the next. Analyzing daily returns provides insight into the stock's volatility and risk, as it reflects the day-to-day fluctuations in the stock's value. By examining these daily changes, we can better understand the stock's behavior and assess its risk profile, beyond just looking at its absolute price.
        df['Daily Return'] = df['Adj Close'].pct_change()
        #fig = px.line(df, x='Date', y='Daily Return', title='Daily Returns in % of NVIDIA Stock', color_discrete_sequence=['#76b900'])
        #fig.show()

        df.to_csv(out_csv, index=False) 
    except ValueError as ve:
        return str(ve)

def train(train_X, train_y, model): 
    try:
        print('--- Training ---')
        automl = AutoML()
        # Specify automl goal and constraint
        automl_settings = {
            "time_budget": 3600,  # in seconds
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
    run_option = 2
    try:
        print("Hello FLAML!")
        logger.setLevel(logging.ERROR)
        csv_file = './datasets/NVDA.csv'
        clean_csv_file = './output/NVDA_cleaned.csv'
        out_model = './output/NVDA.pkl'
        out_result = './output/NVDA.txt'

        match run_option:
            case 0:
                data_analysis(csv_file)                
            case 1:
                data_cleaning(csv_file, clean_csv_file)
            case 2:
                data_cleaning(csv_file, clean_csv_file)
                df = pd.read_csv(clean_csv_file)
                # Convert `Date` column to datetime type
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    # Extract date-related features
                    df['Hour'] = df['Date'].dt.hour
                    df['Day'] = df['Date'].dt.day
                    df['Month'] = df['Date'].dt.month
                    df['Year'] = df['Date'].dt.year
                    # Drop columns
                    df.drop('Date', axis=1, inplace=True)
                # Data split
                print(df.head())
                X = df.drop(['Adj Close', '50_EMA', '200_EMA', '50_MA', '200_MA'], axis=1)
                y = df['Adj Close']
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