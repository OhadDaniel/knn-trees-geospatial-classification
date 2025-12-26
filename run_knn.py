import numpy as np
import pandas as pd
from knn import KNNClassifier
np.random.seed(0)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X_train = train_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_train = train_df["state"].to_numpy()

X_test = test_df[["long", "lat"]].to_numpy(dtype=np.float32)
y_test = test_df["state"].to_numpy()

ks = [1, 10, 100, 1000, 3000]
distance_metrics = ["l1", "l2"]

results = []

for k in ks:
    for metric in distance_metrics:
        print(f"Training KNN with k={k}, metric={metric}")

        clf = KNNClassifier(k=k, distance_metric=metric)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = np.mean(y_pred == y_test)

        results.append({
            "k": k,
            "metric": metric,
            "model": clf,
            "accuracy": accuracy,
        })

        print(f"  accuracy = {accuracy:.4f}")


table = pd.DataFrame(index=ks, columns=distance_metrics, dtype=float)

for r in results:
    k = r["k"]
    metric = r["metric"]
    acc = r["accuracy"]
    table.loc[k, metric] = acc

print("\nAccuracy table (rows = k, columns = metric):")
print(table)