import numpy as np
import pandas as pd
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

# 2. Load XGBoost model using the helper function
from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4)

# 3. Train
xgb_model.fit(X_train, y_train)

# 4. Compute accuracies
train_acc = np.mean(xgb_model.predict(X_train) == y_train)
val_acc = np.mean(xgb_model.predict(X_val) == y_val)
test_acc = np.mean(xgb_model.predict(X_test) == y_test)

print("XGBoost (300 trees, max_depth=6, learning_rate=0.1)")
print(f"  train accuracy = {train_acc:.3f}")
print(f"  val accuracy   = {val_acc:.3f}")
print(f"  test accuracy  = {test_acc:.3f}")

# 5. Visualization
plot_decision_boundaries(
    xgb_model,
    X_test,
    y_test,
    title="XGBoost Decision Boundaries (300 trees, max_depth=6, lr=0.1)"
)
