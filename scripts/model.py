import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
import joblib
import sys

import taxi_driver.taxi_driver as td


def build_pipeline(cat_params, numerical_cols, categorical_cols):
    """
    Build a preprocessing and modeling pipeline for tip prediction.

    Parameters:
    ----------
        cat_params (dict): Parameters for CatBoostRegressor.
        numerical_cols (list): List of numerical feature names.
        categorical_cols (list): List of categorical feature names.

    Returns:
        sklearn.Pipeline: The complete pipeline.
    """
    categorical_transformer = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', CatBoostRegressor(**cat_params))
    ])
    return pipeline

def train_and_evaluate(X:pd.DataFrame,
                       y:pd.DataFrame,
                       pipeline,
                       save_path=None):
    """
    Train the pipeline and evaluate on a holdout set.

    Parameters:
    ----------
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        pipeline (sklearn.Pipeline): The pipeline to train.
        save_path (str, optional): If provided, saves the trained pipeline.

    Returns:
        dict: Evaluation metrics.
    """
    
    # split train /test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)


    # compute mterics
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

    if save_path:
        joblib.dump(pipeline, save_path)

    return metrics


def run_inference(pipeline_path:str,
                  input_path:str,
                  output_csv:str=None):
    """
    Run inference using a stored pipeline on a new dataset.

    Parameters:
    ----------
        pipeline_path (str): Path to the saved pipeline.
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save predictions as CSV.
    """
    pipeline = joblib.load(pipeline_path)
    df = pd.read_parquet(input_path)
    
    # apply needed transformations and feature engineering
    taxi_df=td.Taxidf([input_path],sample_fraction=0.3)
    taxi_df.feature_engineering()
    taxi_df.filter_out_outliers_and_na()
    
    df = taxi_df.taxi_data
    
    # Use the same columns as in training
    numerical_cols = ['fare_amount', 'trip_distance', 'total_amount', 'extra', 'tolls_amount']
    categorical_cols = [
        'pu_day', 'pu_hour', 'Airport_flag', 'congestion_surcharge_flag',
        'is_weekend', 'is_night', 'mta_tax_flag'
    ]
    
    X = df[numerical_cols + categorical_cols]
    preds = pipeline.predict(X)
    df['predicted_tip_amount'] = preds
    
    
    # store in csv
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    else:
        print(df[['predicted_tip_amount']].head())
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi Tip Prediction")
    parser.add_argument('--mode', choices=['train', 'inference'], default='inference', help="Mode: train or inference")
    parser.add_argument('--input', type=str, help="Input CSV or parquet file")
    parser.add_argument('--pipeline', type=str, default="catboost_tip_pipeline.joblib", help="Pipeline path")
    parser.add_argument('--output', type=str, help="Output CSV for predictions (inference mode only)")
    args = parser.parse_args()

    numerical_cols = ['fare_amount', 'trip_distance', 'extra', 'tolls_amount']
    categorical_cols = [
        'PULocationID','DOLocationID','pu_day', 'pu_hour', 'Airport_flag', 'congestion_surcharge_flag',
        'is_weekend', 'is_night', 'mta_tax_flag'
    ]

    if args.mode == 'train':
        if args.input is None:
            print("Please provide --input path to training data (parquet or csv).")
            sys.exit(1)
        if args.input.endswith('.parquet'):
            clean_df = pd.read_parquet(args.input)
        else:
            clean_df = pd.read_csv(args.input)
        X = clean_df[numerical_cols + categorical_cols]
        y = clean_df['tip_amount']

        cat_params = {
            'iterations': 300,
            'learning_rate': 0.1,
            'depth': 4,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.7,
            'verbose': 50,
            'early_stopping_rounds': 30,
            'random_seed': 42,
            'task_type': 'CPU',
            'one_hot_max_size': 2,
        }

        pipeline = build_pipeline(cat_params, numerical_cols, categorical_cols)
        metrics = train_and_evaluate(X, y, pipeline, save_path=args.pipeline)

        print(f"MAE: {metrics['MAE']:.3f}")
        print(f"MSE: {metrics['MSE']:.3f}")
        print(f"RMSE: {metrics['RMSE']:.3f}")
        print(f"R²: {metrics['R2']:.3f}")

    elif args.mode == 'inference':
        if args.input is None:
            print("Please provide --input path to inference data (csv).")
            sys.exit(1)
        run_inference(args.pipeline, args.input, args.output)