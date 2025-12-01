import pandas as pd
from sklearn.preprocessing import StandardScaler

# Choose the features that define similarity between road segments
feature_cols = [
    "crash_count_per_year",
    "severe_share",
    "ped_bike_share",
    "speed_limit",
    "n_lanes",
    "func_arterial",
    "func_freeway",
    "func_local"
]

# Build the feature matrix and handle any missing data
X = data[feature_cols].fillna(0)

# Standardize the features so each contributes on a similar scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
