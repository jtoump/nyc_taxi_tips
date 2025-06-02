import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import networkx as nx
import pandas as pd
import numpy as np

class TaxiGraphDataset:
    def __init__(self, taxi_data: pd.DataFrame):
        self.df = taxi_data.copy()
        self.G = nx.DiGraph()

    def preprocess(self):
        # Build graph: nodes are unique PULocationID and DOLocationID
        all_nodes = pd.unique(self.df[['PULocationID', 'DOLocationID']].values.ravel())
        self.G.add_nodes_from(all_nodes)
        # Edges: from PU to DO, weight is count of trips
        edge_weights = self.df.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='weight')
        for _, row in edge_weights.iterrows():
            self.G.add_edge(row['PULocationID'], row['DOLocationID'], weight=row['weight'])

    def to_pyg_data(self, target_col='tip_amount'):
        # Feature engineering
        # self.df['tip_pct'] = self.df['tip_amount'] / (self.df['fare_amount'] + 1e-3)
        if 'tpep_pickup_datetime' in self.df.columns and 'tpep_dropoff_datetime' in self.df.columns:
            self.df['trip_duration'] = (
                pd.to_datetime(self.df['tpep_dropoff_datetime']) - pd.to_datetime(self.df['tpep_pickup_datetime'])
            ).dt.total_seconds() / 60
            self.df['avg_speed'] = self.df['trip_distance'] / (self.df['trip_duration'] / 60 + 1e-3)
        else:
            self.df['trip_duration'] = 0
            self.df['avg_speed'] = 0

        numerical_cols = [
            'fare_amount', 'trip_distance', 'total_amount', 'extra', 'tolls_amount',
             'trip_duration', 'avg_speed'
        ]
        categorical_cols = [
            'PULocationID','DOLocationID', 'pu_day', 'pu_hour', 'Airport_flag',
            'congestion_surcharge_flag', 'is_weekend', 'is_night', 'mta_tax_flag'
        ]

        # One-hot encode categorical features
        cat_df = pd.get_dummies(self.df[categorical_cols].astype(str), drop_first=True)
        features_df = pd.concat([self.df[numerical_cols], cat_df], axis=1)

        # Aggregate features by PULocationID (mean)
        node_features_df = features_df.groupby(self.df['PULocationID']).mean()
        node_targets = self.df.groupby('PULocationID')[target_col].mean()

        # Align node indices
        zone_ids = sorted(self.G.nodes())
        node_features = node_features_df.reindex(zone_ids).fillna(0).values
        node_targets = node_targets.reindex(zone_ids).fillna(0).values

        x = torch.tensor(StandardScaler().fit_transform(node_features), dtype=torch.float)
        y = torch.tensor(node_targets, dtype=torch.float).unsqueeze(1)

        # Edges
        zone_id_to_idx = {zone_id: idx for idx, zone_id in enumerate(zone_ids)}
        edges = []
        orig_edges = []
        for u, v in self.G.edges():
            if u in zone_id_to_idx and v in zone_id_to_idx:
                edges.append((zone_id_to_idx[u], zone_id_to_idx[v]))
                orig_edges.append((u, v))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([self.G[u][v]['weight'] for u, v in orig_edges], dtype=torch.float).unsqueeze(1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data

class GeoGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=2):
        super().__init__()
        self.gat1 = GATConv(in_channels, 8, heads=heads, dropout=0.2)
        self.gat2 = GATConv(8 * heads, out_channels, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return x

def train_geogat(data, epochs=200, lr=0.01, weight_decay=5e-4, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GeoGAT(in_channels=data.x.shape[1], out_channels=1).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        if verbose and (epoch % 20 == 0 or epoch == epochs-1):
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    return model

def predict_geogat(model, data):
    model.eval()
    with torch.no_grad():
        preds = model(data.to(next(model.parameters()).device)).cpu().numpy().flatten()
    return preds

if __name__ == "__main__":
    # Read the same CSV as in test.py
    taxi_data = pd.read_csv('./data/clean_sample.csv')
    # Optionally sample for speed
    # taxi_data = taxi_data.sample(400000, random_state=42)

    dataset = TaxiGraphDataset(taxi_data)
    dataset.preprocess()
    pyg_data = dataset.to_pyg_data(target_col='tip_amount')

    # Train/test split by node (zone)
    n_nodes = pyg_data.x.shape[0]
    idx = np.arange(n_nodes)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    pyg_data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    pyg_data.test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    pyg_data.train_mask[train_idx] = True
    pyg_data.test_mask[test_idx] = True

    model = train_geogat(pyg_data, epochs=10000, lr=0.0005)

    preds = predict_geogat(model, pyg_data)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = pyg_data.y.squeeze().numpy()
    print("MAE:", mean_absolute_error(y_true[test_idx], preds[test_idx]))
    print("RMSE:", np.sqrt(mean_squared_error(y_true[test_idx], preds[test_idx])))
    print("R2:", r2_score(y_true[test_idx], preds[test_idx]))