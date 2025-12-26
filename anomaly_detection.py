import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from knn import KNNClassifier


# 1. Load data
train_df = pd.read_csv("train.csv")
ad_test_df = pd.read_csv("AD_test.csv")

X_train = train_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_train = train_df["state"].to_numpy()

X_ad = ad_test_df[["long", "lat"]].to_numpy(dtype=np.float32)


# 2. Build kNN model with k=5, L2 distance
model = KNNClassifier(k=5, distance_metric="l2")
model.fit(X_train, y_train)

# 3. Compute kNN distances for each point in AD_test
distances, neighbor_indices = model.knn_distance(X_ad)

# 4. Compute anomaly scores = sum of the 5 distances for each test sample
anomaly_scores = np.sum(distances, axis=1)

# 5. Find indices of the 50 points with the highest anomaly scores
num_anomalies = 50
sorted_indices = np.argsort(anomaly_scores)
anomaly_indices = sorted_indices[-num_anomalies:]
is_anomaly = np.zeros(X_ad.shape[0], dtype=bool)
is_anomaly[anomaly_indices] = True

# 6. Plot:
plt.figure(figsize=(8, 6))


# Train points in black, very low alpha
plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c="black", alpha=0.03, s=2, label="Train (normal region)"
)

# Normal AD_test points in blue
normal_mask = ~is_anomaly
plt.scatter(
    X_ad[normal_mask, 0], X_ad[normal_mask, 1],
    c="blue", alpha=0.8, s=5, label="AD_test normal"
)

# Anomalous AD_test points in red
plt.scatter(
    X_ad[is_anomaly, 0], X_ad[is_anomaly, 1],
    c="red", alpha=0.9, s=5, label="AD_test anomalies"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Anomaly Detection using kNN (k=5, L2)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()