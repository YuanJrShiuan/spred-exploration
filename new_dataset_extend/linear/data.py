import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def isotropic_predictor_data(num_samples, predictor_dim, respond_dim, noisy_variance, sparse=0, seed=666):
    np.random.seed(seed)
    x = np.random.randn(num_samples, predictor_dim)

    trans = np.random.randn(predictor_dim, respond_dim)
    if sparse > 0:
        sparse_mask = np.random.rand(trans.shape)
        trans = (sparse_mask > sparse) * trans

    y = x.dot(trans) + np.random.randn(num_samples,
                                       respond_dim) * noisy_variance

    return (x, y), trans

def isotropic_predictor_data_torch(num_samples, predictor_dim, respond_dim, noisy_variance, sparse=0, seed=666):
    np.random.seed(seed)
    x = np.random.randn(num_samples, predictor_dim)

    trans = np.random.randn(predictor_dim, respond_dim)
    if sparse > 0:
        sparse_mask = np.random.rand(trans.shape)
        trans = (sparse_mask > sparse) * trans

    y = x.dot(trans) + np.random.randn(num_samples,
                                       respond_dim) * noisy_variance

    return (x, y), trans


def load_darwin():
    # Fetch dataset
    df = pd.read_csv("DARWIN.csv")

    # Split features and target
    X = df.drop(columns=["ID", "class"])
    y = df[["class"]]  # Keep as DataFrame for object detection

    # Drop rows with missing values
    valid_idx = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    # Convert features to numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X.dropna(inplace=True, axis=1)  # Drop columns with NaNs

    # Encode categorical target
    if y.select_dtypes(include='object').shape[1] > 0:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.values.ravel())
    else:
        y = y.values.ravel()

    # Reshape y to 2D
    y = y.reshape(-1, 1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return (X_scaled, y), None



import os
import kagglehub

def load_advertise(test_size=0.2, random_state=42):
    # Download dataset
    path = kagglehub.dataset_download("uciml/internet-advertisements-data-set")
    data_file = os.path.join(path, "add.csv")

    # Generate unique feature names
    feature_names = [f"feature_{i}" for i in range(1558 - 1)] + ["target"]

    # Load data
    df = pd.read_csv(data_file, header=None, names=feature_names, na_values="?", low_memory=False)

    # Drop columns with excessive missing values
    df = df.dropna(thresh=len(df) * 0.5, axis=1)
    df = df.dropna()

    # Split
    X = df.drop(columns=["target"])
    y = df["target"].astype(str)
    y = LabelEncoder().fit_transform(y)

    X = X.apply(pd.to_numeric, errors='coerce')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    y = y.reshape(-1, 1)
    # Train-test split
    return (X_scaled, y), None

