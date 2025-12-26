import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from helpers import plot_decision_boundaries

np.random.seed(0)

# Load train + test
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

X_train = train_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_train = train_df["state"].to_numpy()

X_test  = test_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_test  = test_df["state"].to_numpy()

# Build the selected tree (max_depth=100, max_leaf_nodes=50)
tree_50 = DecisionTreeClassifier(
    max_depth=100,
    max_leaf_nodes=50,
    random_state=0
)

tree_50.fit(X_train, y_train)

# Plot decision boundaries
plot_decision_boundaries(
    tree_50,
    X_test,
    y_test,
    title="Decision Tree with 50 Leaf Nodes (max_depth=100)"
)