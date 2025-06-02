




## Quick Installation

1. Ensure you have installed uv else `pip install uv`

2. Then run `uv sync`  in the rood directory

3. Install the module that was created for the test

```bash
#activate the venv that was created through uv
source .venv/bin/activate

#cd to th source directory and install the setup.py
cd srs
pip install -e .
```

## Data modeling Notebooks

- [Data Processing](./notebooks/Tip_Prediction_2025.ipynb)
- [Linear Models](./notebooks/Linear_Models.ipynb)
- [Boosting Models](./notebooks/Boosting_Models.ipynb)


## Scripts


```bash

#train caboost
 python scripts/model.py --mode train --input ./data/clean_sample.parquet --pipeline ./data/models/pipeline_model.joblib --output results.csv 

```
```bash
#infer caboost from saved pipeline

 python scripts/model.py --mode inference --input ./data/clean_sample.parquet --pipeline ./data/models/pipeline_model.joblib --output results.csv

```
