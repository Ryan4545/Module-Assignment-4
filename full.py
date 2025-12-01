import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the data
exec(open("data_generations.py").read())

# === 1. Build feature matrix ===

feature_cols = [
    "crash_count_per_year",
    "severe_share",
    "ped_bike_share",
    "speeding_share",
    "speed_limit",
    "n_lanes",
    "bike_lanes",
    "traffic_volume"
]

# Only use rows where we have roadway data
data_clean = data.dropna(subset=["speed_limit", "n_lanes"])
X = data_clean[feature_cols].fillna(0)

# === 2. Standardize features ===

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Explore different values of k ===

wcss = []
sil_scores = []
ks = range(2, 11)

for k in ks:
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Elbow plot
plt.figure()
plt.plot(ks, wcss, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Within-cluster sum of squares (WCSS)")
plt.title("Elbow Plot for K-Means on DC Roadway Segments")
plt.tight_layout()
plt.savefig("figure_1_elbow_plot.png", dpi=300)

# Silhouette plot
plt.figure()
plt.plot(ks, sil_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Scores for Different Values of k")
plt.tight_layout()
plt.savefig("figure_2_silhouette_scores.png", dpi=300)

# === 4. Final K-Means model with chosen k ===

k_final = 4
kmeans = KMeans(n_clusters=k_final, n_init=20, random_state=42)
data_clean["cluster"] = kmeans.fit_predict(X_scaled)

# === 5. Inspect cluster characteristics ===

cluster_summary = data_clean.groupby("cluster")[feature_cols].mean()
print("Cluster summary (mean feature values):")
print(cluster_summary)

# === 6. Simple 2D visualization ===

plt.figure()
scatter = plt.scatter(
    data_clean["crash_count_per_year"],
    data_clean["ped_bike_share"],
    c=data_clean["cluster"],
    alpha=0.6
)
plt.xlabel("Crashes per year")
plt.ylabel("Ped/Bike crash share")
plt.title("DC Roadway Segments Clustered by Crash Risk and Mode Mix")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("figure_3_cluster_scatter.png", dpi=300)
