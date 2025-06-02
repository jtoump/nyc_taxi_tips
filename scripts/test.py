
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from catboost import CatBoostRegressor	
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


clean_df=pd.read_csv('./data/clean_sample.csv')

clean_df = clean_df.sample(400000)

# # Intlize Power Transformer
# pt = PowerTransformer(standardize=True, method='yeo-johnson')

# # Numerical Columsn to Transform
# cols_num_pt = ['tolls_amount', 'extra']

# # Fit and Transform
# for col in cols_num_pt:
#     # Fit and transform
#     clean_df[col] = pt.fit_transform(clean_df[[col]])

# # Intlize Power Transformer
# qt = QuantileTransformer(output_distribution='normal')

# # Numerical Columsn to Transform 
# cols_num = ['trip_distance', 'fare_amount','total_amount']

# # Fit and Transform
# for col in cols_num:
#     # Fit and transform
#     clean_df[col] = qt.fit_transform(clean_df[[col]])
    
    
numerical_cols=['fare_amount','trip_distance','total_amount','extra','tolls_amount']


categorical_cols = ['PULocationID',
                    'DOLocationID',
                    'pu_day',
                    'pu_hour',
                    'Airport_flag',
                    'congestion_surcharge_flag',
                    'is_weekend',
                    'is_night',
                    'mta_tax_flag'
]
    
    
X=clean_df[numerical_cols+categorical_cols]
y=clean_df['tip_amount']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


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

# No imputer here
categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Pass-through numerical features as-is (no imputation)
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

catBoost = Pipeline([
    ('preprocess', preprocessor),
    ('model', CatBoostRegressor(**cat_params))
])

catBoost.fit(X_train, y_train)
y_pred_xgb = catBoost.predict(X_test)


# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred_xgb)
mse = mean_squared_error(y_test, y_pred_xgb)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_xgb)

# Print metrics
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

