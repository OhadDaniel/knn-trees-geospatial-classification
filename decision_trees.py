import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
np.random.seed(0)
# 1. Load data
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

# train
X_train = train_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_train = train_df["state"].to_numpy()

# validation
X_val = val_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_val = val_df["state"].to_numpy()

# test
X_test = test_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_test = test_df["state"].to_numpy()

# 2. Hyperparameters
max_depth_values = [1, 2, 4, 6, 10, 20, 50, 100]
max_leaf_nodes_values = [50, 100, 1000]

results = []
i = 1
# 3. Train 24 trees
for max_depth in max_depth_values:
    for max_leaf_nodes in max_leaf_nodes_values:

        print(f"Training tree {i}: max_depth={max_depth}, max_leaf_nodes={max_leaf_nodes}")

        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            random_state=0,
            )

        tree.fit(X_train, y_train)

        # accuracies
        train_acc = np.mean(tree.predict(X_train) == y_train)
        val_acc = np.mean(tree.predict(X_val) == y_val)
        test_acc = np.mean(tree.predict(X_test) == y_test)

        results.append({
            "max_depth": max_depth,
            "max_leaf_nodes": max_leaf_nodes,
            "model": tree,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
        })

        print(f"  train={train_acc:.3f}, val={val_acc:.3f}, test={test_acc:.3f}")
        i += 1