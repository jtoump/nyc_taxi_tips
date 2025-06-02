import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from typing import List, Optional, Union


class TaxiDF:
    def __init__(self, src_paths: Union[str, List[str]], 
                 taxi_zones_shp_path: str = "", 
                 sample_fraction: Optional[float] = None):
        """
        Load NYC taxi trip data from one or more parquet files.
        """
        self.taxi_data = pd.DataFrame()
        self.taxi_zones_shp_path = taxi_zones_shp_path

        if isinstance(src_paths, str):
            src_paths = [src_paths]
        
        for path in src_paths:
            df = pd.read_parquet(path)
            if sample_fraction:
                df = df.sample(frac=sample_fraction)
            self.taxi_data = pd.concat([self.taxi_data, df], ignore_index=True)

        self.convert_dates()

    def convert_dates(self):
        """Extract useful time features from datetime columns."""
        self.taxi_data['tpep_pickup_datetime'] = pd.to_datetime(self.taxi_data['tpep_pickup_datetime'])
        self.taxi_data['tpep_dropoff_datetime'] = pd.to_datetime(self.taxi_data['tpep_dropoff_datetime'])
        self.taxi_data['pu_hour'] = self.taxi_data['tpep_pickup_datetime'].dt.hour
        self.taxi_data['pu_day'] = self.taxi_data['tpep_pickup_datetime'].dt.dayofweek
        self.taxi_data['pu_date'] = self.taxi_data['tpep_pickup_datetime'].dt.date

    def compute_taxi_zones(self):
        """Attach pickup/dropoff counts to taxi zones."""
        self.taxi_zones = gpd.read_file(self.taxi_zones_shp_path)

        for col in ['PULocationID', 'DOLocationID']:
            zone_counts = self.taxi_data.groupby(col).size().reset_index(name=f"{col}_count")
            self.taxi_zones = self.taxi_zones.merge(zone_counts, left_on="LocationID", right_on=col, how='left')
        
        self.taxi_zones.fillna(0, inplace=True)
        self.taxi_zones['taxi_demand'] = self.taxi_zones['PULocationID_count'] - self.taxi_zones['DOLocationID_count']
        self.taxi_zones['taxi_demand_class'] = (self.taxi_zones['taxi_demand'] > 0).astype(int)

    def describe_by_group(self, group_col='payment_type', value_col='tip_amount'):
        """Group by and describe a column."""
        return self.taxi_data.groupby(group_col)[value_col].describe().reset_index()

    def plot_hourly_distribution(self):
        sns.histplot(self.taxi_data['pu_hour'], bins=24, kde=True)
        plt.title("Pickup Hour Distribution")
        plt.xlabel("Hour")
        plt.ylabel("Count")
        plt.show()

    def demand_prediction_model(self):
        """
        Train a model to predict total_amount from pickup features.
        """
        features = ['trip_distance', 'passenger_count', 'pu_hour', 'pu_day']
        df = self.taxi_data.dropna(subset=features + ['total_amount'])
        X = df[features]
        y = df['total_amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        print(f"Model R² score: {score:.4f}")
        return model

    def cluster_zones(self, k: int = 10):
        """
        Cluster zones based on pickup and dropoff location frequency.
        """
        coords = self.taxi_zones.copy()
        coords['centroid_x'] = coords.geometry.centroid.x
        coords['centroid_y'] = coords.geometry.centroid.y
        kmeans = KMeans(n_clusters=k, random_state=42)
        coords['cluster'] = kmeans.fit_predict(coords[['centroid_x', 'centroid_y']])
        self.taxi_zones['cluster'] = coords['cluster']
        return coords

    def build_graph(self):
        """
        Build a NetworkX graph from OD pairs.
        """
        self.graph = nx.DiGraph()
        for _, row in self.taxi_data.iterrows():
            u = row['PULocationID']
            v = row['DOLocationID']
            if self.graph.has_edge(u, v):
                self.graph[u][v]['weight'] += 1
            else:
                self.graph.add_edge(u, v, weight=1)
        print(f"Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        return self.graph
