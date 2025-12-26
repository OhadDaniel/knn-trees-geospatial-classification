
import numpy as np

np.random.seed(0)

class KNNClassifier:
    def __init__(self, k, distance_metric='l2'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.Y_train = None

    def fit(self, X_train, Y_train):
        import faiss
        """
        Update the kNN classifier with the provided training data.

        Parameters:
        - X_train (numpy array) of size (N, d): Training feature vectors.
        - Y_train (numpy array) of size (N,): Corresponding class labels.
        """
        self.X_train = X_train.astype(np.float32)
        self.Y_train = Y_train
        d = self.X_train.shape[1]
        if self.distance_metric == 'l2':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
        elif self.distance_metric == 'l1':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L1)
        else:
            raise NotImplementedError
        pass
        self.index.add(self.X_train)

    def predict(self, X):

        distances, neighbor_indices = self.knn_distance(X)
        neighbor_labels = self.Y_train[neighbor_indices]

        num_samples = neighbor_labels.shape[0]
        predicted_labels = np.empty(num_samples, dtype=self.Y_train.dtype)

        for i in range(num_samples):
            labels_i, counts_i = np.unique(neighbor_labels[i], return_counts=True)
            majority_label = labels_i[np.argmax(counts_i)]
            predicted_labels[i] = majority_label

        return predicted_labels

    def knn_distance(self, X):

        X = X.astype(np.float32)
        D, I = self.index.search(X, self.k)
        return D, I
