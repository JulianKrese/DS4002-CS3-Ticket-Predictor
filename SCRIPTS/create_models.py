"""
===============================================================================
File: create_models.py
Description:
    This script trains Isolation Forest models on cleaned and encoded parking
    ticket data. The dataset is split into multiple time frames for training,
    including all-time, recent year, and 5-year descending chunks. Each model
    is trained to detect anomalies (rare/low-probability tickets) using only
    positive examples. The script scales features, trains the model, and saves
    both the model and its scaler for future inference.
Usage:
    python create_models.py
Inputs:
    - ../DATA/Final/encoded_parking_tickets.csv : preprocessed numeric dataset
Outputs:
    - ../OUTPUT/models/ : directory containing trained models and scalers
        e.g. isolation_forest_2010-2014.joblib
             scaler_2010-2014.joblib
===============================================================================
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import os
import shutil

# -------------------------------
# 1. Load encoded parking ticket data
# -------------------------------
data_file = "./DATA/Final/encoded_parking_tickets.csv"
df = pd.read_csv(data_file)

# -------------------------------
# 2. Define features for modeling
# -------------------------------
# Use all columns except 'Year' as features
features = df.columns.tolist()
features.remove('Year')
X_features = features.copy()  # list of feature names

# -------------------------------
# 3. Create output directory for models
# -------------------------------
base_dir = "./OUTPUT"
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir)

model_dir = "./OUTPUT/models"
# Remove old models directory if it exists, to avoid conflicts
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

# Recreate the directory
os.makedirs(model_dir)

# -------------------------------
# 4. Split data into time frames
# -------------------------------
def get_time_splits(df, start_year=2000, end_year=2023, chunk=5):
    """
    Splits the dataset into multiple time frames for training.
    
    Parameters:
        df (pd.DataFrame): encoded parking ticket dataset
        start_year (int): earliest year to consider
        end_year (int): latest year to consider for training
        chunk (int): size of year chunks (e.g., 5-year chunks)
    
    Returns:
        splits (dict): dictionary of training splits labeled by timeframe
        test_df (pd.DataFrame): test set containing only 2024 data
    """

    splits = {}

    # 4a. All-time split (everything up to end_year)
    splits['all_time'] = df[df['Year'] <= end_year]

    # 4b. Recent year split (e.g., 2023)
    splits['recent'] = df[df['Year'] == 2023]

    # 4c. 5-year chunks, descending
    for y_end in range(end_year, start_year - 1, -chunk):
        y_start = max(y_end - chunk + 1, start_year)  # ensure start_year is respected
        label = f"{y_start}-{y_end}"
        df_split = df[(df['Year'] >= y_start) & (df['Year'] <= y_end)]
        # Only include splits with enough rows for training
        if len(df_split) > 2500:
            splits[label] = df_split

    # 4d. Test set: all data from 2024
    test_df = df[df['Year'] == 2024]

    return splits, test_df

# Generate splits and test set
splits, test_df = get_time_splits(df, start_year=df['Year'].min(), end_year=2023, chunk=5)

# -------------------------------
# 5. Train Isolation Forest models for each split
# -------------------------------
for label, split_df in splits.items():

    # Skip empty splits
    if split_df.empty:
        print(f"Skipping {label} — no data.")
        continue

    print(f"Training Isolation Forest for {label} ({len(split_df)} rows)...")

    # Select features for training
    X = split_df[X_features]

    # -------------------------------
    # 5a. Scale features
    # -------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------------
    # 5b. Train Isolation Forest
    # -------------------------------
    # n_estimators: number of trees in the forest
    # contamination: proportion of anomalies expected (used for thresholding)
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_scaled)

    # -------------------------------
    # 5c. Save trained model and scaler
    # -------------------------------
    model_file = f"{model_dir}/isolation_forest_{label}.joblib"
    scaler_file = f"{model_dir}/scaler_{label}.joblib"

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    print(f"Saved model to {model_file}")
    print(f"Saved scaler to {scaler_file}")
