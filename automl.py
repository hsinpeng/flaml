import sys
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pickle
from flaml import AutoML
from flaml import logger

def train(train, model): 
    try:
        automl = AutoML()
        # Specify automl goal and constraint
        automl_settings01 = {
            "time_budget": 300,  # in seconds
            #"metric": "roc_auc",
            "task": "classification",
            "estimator_list":['lgbm', 'rf', 'extra_tree', 'lrl1'],
            "log_file_name": "churn.log",
        }

        automl_settings02 = {
            "time_budget": 3600,  # total running time in seconds (1hr)
            "metric": 'roc_auc', 
            "task": 'classification',  # task type
            #"estimator_list":['lgbm', 'rf', 'extra_tree', 'xgboost', 'xgb_limitdepth', 'lrl1', 'catboost'],
            "estimator_list":['lgbm', 'rf', 'extra_tree', 'xgb_limitdepth', 'lrl1'],
            "log_file_name": 'churn.log',
            "log_training_metric": True,  # whether to log training metric
            "keep_search_state": True, # needed if you want to keep the cross validation information
            "eval_method": "cv",
            "split_type": "stratified",
            "n_splits": 5,
            "ensemble":{
                # final model will be a stacked ensemble of the best model for each estimator type
                "final_estimator": LogisticRegression(),
                "passthrough": False
            },
            "log_type":'all'
        }

        automl.fit(train.drop('Exited', axis=1), train['Exited'], **automl_settings01)

        # Save model
        with open(model, "wb") as f:
            pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

        # Best loss per estimator
        best_loss = (1-pd.Series(automl.best_loss_per_estimator)).sort_values(ascending=False).round(4)
        print(best_loss)

        # Best configuration per estimator
        best_configuration = automl.best_config_per_estimator
        print(best_configuration)

    except ValueError as ve:
        return str(ve)

def test(model, test, result):
    try:
        with open(model, "rb") as f:
            automl = pickle.load(f)

        # Generate predictions on test set and save submission file
        submission = pd.DataFrame({
            'id': test.index.tolist(),
            'Exited': automl.predict_proba(test)[:, 1]
        })

        submission.to_csv(result, index=False)
    except ValueError as ve:
        return str(ve)

def main():
    run_option = 1
    try:
        print("Hello FLAML!")
        logger.setLevel(logging.ERROR)

        out_model = './output/automl.pkl'
        out_csv = './output/submission.csv'
        train_csv = pd.read_csv('./datasets/train.csv', index_col='id').drop(['CustomerId', 'Surname'], axis=1)
        test_csv = pd.read_csv('./datasets/test.csv', index_col='id').drop(['CustomerId', 'Surname'], axis=1)

        match run_option:
            case 0:
                train(train_csv, out_model)
            case 1:
                test(out_model, test_csv, out_csv)
            case _:
                print(f'Error: Wrong run_option:{run_option}!')

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())