# figures.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# === 0. Load your prepared segment-level dataset ===
# If you already saved the aggregated data to CSV, load it here.
# Otherwise, import it from another module if you have it in memory elsewhere.
data = pd.read_csv("segments_prepared.csv")  # <-- change this filename

# === 1. Build feature matrix (same as in your clustering script) ===

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

X = data[feature_cols].fillna(0)

# === 2. Standardize ===

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Compute WCSS and silhouette scores for different k ===

wcss = []
sil_scores = []
ks = range(2, 11)

for k in ks:
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# === 4. Elbow plot ===

plt.figure()
plt.plot(list(ks), wcss, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Within-cluster sum of squares (WCSS)")
plt.title("Elbow Plot for K-Means on DC Roadway Segments")
plt.tight_layout()
plt.savefig("figure_1_elbow_plot.png", dpi=300)

# === 5. Silhouette plot ===

plt.figure()
plt.plot(list(ks), sil_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Scores for Different Values of k")
plt.tight_layout()
plt.savefig("figure_2_silhouette_scores.png", dpi=300)

# === 6. Optional: 2D scatter using your final k (e.g., 4) ===

k_final = 4
kmeans_final = KMeans(n_clusters=k_final, n_init=20, random_state=42)
data["cluster"] = kmeans_final.fit_predict(X_scaled)

plt.figure()
scatter = plt.scatter(
    data["crash_count_per_year"],
    data["ped_bike_share"],
    c=data["cluster"],
    alpha=0.6
)
plt.xlabel("Crashes per year")
plt.ylabel("Ped/Bike crash share")
plt.title("DC Roadway Segments Clustered by Crash Risk and Mode Mix")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("figure_3_cluster_scatter.png", dpi=300)
