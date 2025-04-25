import GEOparse
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

def get_cancer_GDS(filepath):
    gds = GEOparse.get_GEO(filepath=filepath)
    X = []
    y = []
    subset_keys = list(gds.subsets.keys())
    for i, k in enumerate(subset_keys):
        sample_ids = gds.subsets[k].metadata['sample_id'][0].split(',')
        for sample_id in sample_ids:
            _x = gds.table.loc[:, sample_id].to_numpy().reshape((1, -1))
            _y = i
            X.append(_x)
            y.append(_y)

    from collections import Counter
    yc = Counter(y)
    for k, c in yc.most_common(1):
        logging.info(f"most common label {k}: percent {c / len(y)}")

    X_arr = np.concatenate(X, axis=0).astype('float32')
    X_arr = np.nan_to_num(X_arr, nan=0)
    y_arr = np.asarray(y).astype('int64')

    X_arr_mean = np.mean(X_arr, axis=0)
    X_arr_std = np.std(X_arr)
    X_arr -= X_arr_mean
    X_arr /= (X_arr_std + 1e-10)

    logging.info(f"GDS dataset {filepath} loaded")
    logging.info(f"#features {X_arr.shape[1]}, #labels {np.max(y_arr)+1}, #samples {X_arr.shape[0]}")
    return X_arr, y_arr

def get_darwin(test_size=0.2, random_state=42):
    # Fetch the dataset
    darwin = fetch_ucirepo(id=732)

    # Extract and clean data
    X = darwin.data.features
    y = darwin.data.targets

    # Remove rows with missing data
    valid_idx = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    # Convert all features to numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X.dropna(inplace=True, axis=1)

    # Encode categorical target, if needed
    if y.dtypes[0] == 'object':
        y = LabelEncoder().fit_transform(y.values.ravel())
    else:
        y = y.values.ravel()

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    return X_scaled,y


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
    # Train-test split
    return X_scaled,y