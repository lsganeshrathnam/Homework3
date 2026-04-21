import numpy as np
from collections import Counter
import time

# -----------------------------
# Distance Functions
# -----------------------------
def euclidean_distance_matrix(X, centroids):
    # (n_samples, K)
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

def cosine_distance_matrix(X, centroids):
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    C_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    return 1 - np.dot(X_norm, C_norm.T)

def jaccard_distance_matrix(X, centroids):
    distances = np.zeros((X.shape[0], centroids.shape[0]))

    for i, c in enumerate(centroids):
        min_sum = np.minimum(X, c).sum(axis=1)
        max_sum = np.maximum(X, c).sum(axis=1) + 1e-10
        distances[:, i] = 1 - (min_sum / max_sum)

    return distances

# -----------------------------
# K-Means Class
# -----------------------------
class KMeansScratch:
    def __init__(self, K=10, max_iters=100, tol=1e-4, distance='euclidean'):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.distance_type = distance

    def _compute_distances(self, X, centroids):
        if self.distance_type == 'euclidean':
            return euclidean_distance_matrix(X, centroids)
        elif self.distance_type == 'cosine':
            return cosine_distance_matrix(X, centroids)
        elif self.distance_type == 'jaccard':
            return jaccard_distance_matrix(X, centroids)

    def fit(self, X):
        np.random.seed(42)
        self.centroids = X[np.random.choice(len(X), self.K, replace=False)]
        prev_sse = float('inf')

        for iteration in range(self.max_iters):

            # Assign clusters (vectorized)
            distances = self._compute_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = []
            for k in range(self.K):
                points = X[labels == k]
                if len(points) > 0:
                    new_centroids.append(points.mean(axis=0))
                else:
                    new_centroids.append(self.centroids[k])
            new_centroids = np.array(new_centroids)

            # Compute SSE
            min_distances = np.min(distances, axis=1)
            sse = np.sum(min_distances ** 2)

            # Stopping conditions
            centroid_shift = np.linalg.norm(self.centroids - new_centroids)

            if centroid_shift < self.tol:
                break
            if sse > prev_sse:
                break

            self.centroids = new_centroids
            prev_sse = sse

        self.labels_ = labels
        self.sse = sse
        self.iterations = iteration + 1

    def predict(self, X):
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

# -----------------------------
# Accuracy Calculation
# -----------------------------
def compute_accuracy(y_true, cluster_labels):
    label_map = {}

    for cluster in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster)[0]
        true_labels = y_true[indices]
        majority_label = Counter(true_labels).most_common(1)[0][0]
        label_map[cluster] = majority_label

    predicted = np.array([label_map[c] for c in cluster_labels])
    return np.mean(predicted == y_true)

# -----------------------------
# Run Experiment
# -----------------------------
def run_kmeans(X, y, K):
    results = {}

    # Normalize once for cosine
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

    for dist in ['euclidean', 'cosine', 'jaccard']:
        print(f"\nRunning K-Means with {dist}")

        if dist == 'cosine':
            X_used = X_norm
        else:
            X_used = X

        model = KMeansScratch(K=K, max_iters=100, distance=dist)

        start = time.time()
        model.fit(X_used)
        end = time.time()

        labels = model.labels_
        acc = compute_accuracy(y, labels)

        results[dist] = {
            'SSE': model.sse,
            'Accuracy': acc,
            'Iterations': model.iterations,
            'Time': end - start
        }

    return results

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    X = np.loadtxt("data.csv", delimiter=",")
    y = np.loadtxt("label.csv", delimiter=",").astype(int)

    K = len(np.unique(y))

    results = run_kmeans(X, y, K)

    print("\nFinal Results:")
    for method, res in results.items():
        print(f"\n{method.upper()}")
        print(f"SSE: {res['SSE']}")
        print(f"Accuracy: {res['Accuracy']:.4f}")
        print(f"Iterations: {res['Iterations']}")
        print(f"Time: {res['Time']:.2f} sec")