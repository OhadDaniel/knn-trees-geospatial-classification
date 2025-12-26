import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from helpers import plot_decision_boundaries

np.random.seed(0)

# 1. Load data
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

X_train = train_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_train = train_df["state"].to_numpy()

X_test = test_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_test = test_df["state"].to_numpy()

# 2. Recreate the best tree from Q1
best_tree = DecisionTreeClassifier(
    max_depth=20,          # or 50 / 100 – gives the same result with 1000 leaves
    max_leaf_nodes=1000,
    random_state=0,
)

best_tree.fit(X_train, y_train)

# 3. Visualize decision boundaries on the test data
plot_decision_boundaries(
    best_tree,
    X_test,
    y_test,
    title="Decision Tree Decision Boundaries (max_depth=20, max_leaf_nodes=1000)"
)

