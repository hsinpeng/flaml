import sys
import logging
import pandas as pd
from pathlib import Path
import pickle
from flaml import AutoML
from flaml import logger
import seaborn as sns
import matplotlib.pyplot as plt

def data_analysis(csv):
    try:
        print('--- Data Analysis ---')
        df = pd.read_csv(csv) 
    except ValueError as ve:
        return str(ve)   

def data_cleaning(csv, out_csv):
    try:
        print('--- Data Cleaning ---')
        df = pd.read_csv(csv) 
    except ValueError as ve:
        return str(ve) 
         
def train(train, model): 
    try:
        print('--- Training ---')
    except ValueError as ve:
        return str(ve)

def test(model, test, result):
    try:
        print('--- Testing ---')
    except ValueError as ve:
        return str(ve)
def main():
    run_option = 0
    try:
        print("Hello FLAML!")
        logger.setLevel(logging.ERROR)

        match run_option:
            case 0:
                print(f'run_option={run_option}')
            case 1:
                print(f'run_option={run_option}')
            case _:
                print(f'run_option={run_option}')
                print(f'Error: Wrong run_option!')

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())