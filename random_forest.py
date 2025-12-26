import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from helpers import plot_decision_boundaries

np.random.seed(0)

# 1. Load data
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

X_train = train_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_train = train_df["state"].to_numpy()

X_val = val_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_val = val_df["state"].to_numpy()

X_test = test_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_test = test_df["state"].to_numpy()

# 2. Build the random forest: 300 trees, max depth 6
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    n_jobs=4,
    random_state=0,
)

# 3. Train
rf_model.fit(X_train, y_train)

# 4. Compute accuracies
train_acc = np.mean(rf_model.predict(X_train) == y_train)
val_acc = np.mean(rf_model.predict(X_val) == y_val)
test_acc = np.mean(rf_model.predict(X_test) == y_test)

print(f"Random Forest (300 trees, max_depth=6)")
print(f"  train accuracy = {train_acc:.3f}")
print(f"  val accuracy   = {val_acc:.3f}")
print(f"  test accuracy  = {test_acc:.3f}")

# 5. Visualize decision boundaries on the test set
plot_decision_boundaries(
    rf_model,
    X_test,
    y_test,
    title="Random Forest (300 trees, max_depth=6)"
)
