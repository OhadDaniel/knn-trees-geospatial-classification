from knn import KNNClassifier
import pandas as pd
import numpy as np
from helpers import plot_decision_boundaries
np.random.seed(0)

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X_train = test_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_train = test_df["state"].to_numpy()

X_test = test_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_test = test_df["state"].to_numpy()


# --- Model (i): L2, kmax ---
model1 = KNNClassifier(k=1, distance_metric="l2")
model1.fit(X_train, y_train)
plot_decision_boundaries(model1,X_test, y_test,
                         title="KNN Decision Boundary — L2, k=1")

# --- Model (ii): L2, kmin ---
model2 = KNNClassifier(k=3000, distance_metric="l2")
model2.fit(X_train, y_train)
plot_decision_boundaries(model2,X_test, y_test,
                         title="KNN Decision Boundary — L2, k=3000")

# --- Model (iii): L1, kmax ---
model3 = KNNClassifier(k=1, distance_metric="l1")
model3.fit(X_train, y_train)
plot_decision_boundaries(model3, X_test, y_test,
                         title="KNN Decision Boundary — L1, k=1")