import os
from threadpoolctl import threadpool_limits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Fix for KMeans memory leak on Windows
os.environ["OMP_NUM_THREADS"] = "1"

# Load dataset
df = pd.read_csv(r"C:\Users\elz00\Downloads\customer_segmentation.csv")

# Standardize features for better clustering performance
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Apply K-Means with the optimal number of clusters (e.g., k=4)
with threadpool_limits(limits=2, user_api='blas'):
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualizing the clusters (using first two features for 2D representation)
plt.figure(figsize=(8, 5))
plt.scatter(df['Age'], df['Annual Income'], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.title('Customer Segmentation')
plt.colorbar(label='Cluster')
plt.show()

# Save the clustered dataset
df.to_csv("customer_segmented.csv", index=False)
