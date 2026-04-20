import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("C:/Users/ADMIN/Desktop/ratings_small.csv")
user_item = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
matrix = user_item.values

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(pred, true):
    mask = true > 0
    pred = pred[mask]
    true = true[mask]
    pred = np.nan_to_num(pred)

    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    return rmse, mae

# -----------------------------
# USER-CF
# -----------------------------
def user_cf(matrix):
    counts = (matrix != 0).sum(axis=1)
    counts[counts == 0] = 1
    user_means = matrix.sum(axis=1) / counts

    matrix_centered = matrix - user_means[:, np.newaxis]
    matrix_centered[matrix == 0] = 0

    sim = cosine_similarity(matrix_centered)
    denom = np.abs(sim).sum(axis=1, keepdims=True)
    denom[denom == 0] = 1e-10

    pred = user_means[:, np.newaxis] + sim.dot(matrix_centered) / denom
    return np.nan_to_num(pred)

# -----------------------------
# ITEM-CF
# -----------------------------
def item_cf(matrix):
    counts = (matrix != 0).sum(axis=0)
    counts[counts == 0] = 1
    item_means = matrix.sum(axis=0) / counts

    matrix_centered = matrix - item_means
    matrix_centered[matrix == 0] = 0

    sim = cosine_similarity(matrix_centered.T)
    denom = np.abs(sim).sum(axis=1)
    denom[denom == 0] = 1e-10

    pred = item_means + matrix_centered.dot(sim) / denom
    return np.nan_to_num(pred)

# -----------------------------
# PMF (SVD)
# -----------------------------
def pmf(matrix, k=20):
    global_mean = matrix[matrix > 0].mean()

    U, sigma, Vt = np.linalg.svd(matrix, full_matrices=False)
    sigma = np.diag(sigma[:k])
    U = U[:, :k]
    Vt = Vt[:k, :]

    pred = np.dot(U, np.dot(sigma, Vt))
    return np.nan_to_num(pred + global_mean)

# -----------------------------
# Similarities
# -----------------------------
def pearson_similarity(matrix):
    return np.nan_to_num(np.corrcoef(matrix))

def msd_similarity(matrix):
    n = matrix.shape[0]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = matrix[i] - matrix[j]
            sim[i, j] = 1 / (1 + np.mean(diff**2))
    return sim

def user_cf_with_sim(matrix, sim):
    counts = (matrix != 0).sum(axis=1)
    counts[counts == 0] = 1
    user_means = matrix.sum(axis=1) / counts

    matrix_centered = matrix - user_means[:, np.newaxis]
    matrix_centered[matrix == 0] = 0

    denom = np.abs(sim).sum(axis=1, keepdims=True)
    denom[denom == 0] = 1e-10

    pred = user_means[:, np.newaxis] + sim.dot(matrix_centered) / denom
    return np.nan_to_num(pred)

# -----------------------------
# KNN with K neighbors
# -----------------------------
def user_cf_k(matrix, k):
    counts = (matrix != 0).sum(axis=1)
    counts[counts == 0] = 1
    user_means = matrix.sum(axis=1) / counts

    matrix_centered = matrix - user_means[:, np.newaxis]
    matrix_centered[matrix == 0] = 0

    sim = cosine_similarity(matrix_centered)
    pred = np.zeros(matrix.shape)

    for i in range(matrix.shape[0]):
        top_k = np.argsort(sim[i])[-k:]
        sim_k = sim[i, top_k]

        denom = np.sum(np.abs(sim_k))
        if denom == 0:
            denom = 1e-10

        pred[i] = user_means[i] + sim_k.dot(matrix_centered[top_k]) / denom

    return np.nan_to_num(pred)

# -----------------------------
# CROSS VALIDATION
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rows, cols = matrix.nonzero()
ratings = list(zip(rows, cols))

user_rmse, user_mae = [], []
item_rmse, item_mae = [], []
pmf_rmse, pmf_mae = [], []

# -----------------------------
# MODEL EVALUATION
# -----------------------------
for train_idx, test_idx in kf.split(ratings):

    train = matrix.copy()
    test = np.zeros(matrix.shape)

    for idx in test_idx:
        r, c = ratings[idx]
        train[r, c] = 0
        test[r, c] = matrix[r, c]

    # User CF
    pred_user = user_cf(train)
    rmse, mae = evaluate(pred_user, test)
    user_rmse.append(rmse)
    user_mae.append(mae)

    # Item CF
    pred_item = item_cf(train)
    rmse, mae = evaluate(pred_item, test)
    item_rmse.append(rmse)
    item_mae.append(mae)

    # PMF
    pred_pmf = pmf(train)
    rmse, mae = evaluate(pred_pmf, test)
    pmf_rmse.append(rmse)
    pmf_mae.append(mae)

print("\n===== FINAL RESULTS =====")
print("\nUser-CF RMSE:", round(np.mean(user_rmse),4))
print("Item-CF RMSE:", round(np.mean(item_rmse),4))
print("PMF RMSE:", round(np.mean(pmf_rmse),4))

# -----------------------------
# SIMILARITY ANALYSIS + PLOT
# -----------------------------
similarities = {
    "Cosine": lambda x: cosine_similarity(x),
    "Pearson": pearson_similarity,
    "MSD": msd_similarity
}

sim_labels = []
sim_values = []

for name, func in similarities.items():
    rmses = []

    for train_idx, test_idx in kf.split(ratings):
        train = matrix.copy()
        test = np.zeros(matrix.shape)

        for idx in test_idx:
            r, c = ratings[idx]
            train[r, c] = 0
            test[r, c] = matrix[r, c]

        sim = func(train)
        pred = user_cf_with_sim(train, sim)
        rmse, _ = evaluate(pred, test)
        rmses.append(rmse)

    sim_labels.append(name)
    sim_values.append(np.mean(rmses))

plt.figure()
plt.plot(sim_labels, sim_values, marker='o')
plt.title("Similarity vs RMSE (User-CF)")
plt.xlabel("Similarity")
plt.ylabel("RMSE")
plt.grid()
plt.savefig("similarity_plot.png")

# -----------------------------
# K ANALYSIS + PLOT
# -----------------------------
k_values = [5, 10, 20, 50]
k_results = []

for k in k_values:
    rmses = []

    for train_idx, test_idx in kf.split(ratings):
        train = matrix.copy()
        test = np.zeros(matrix.shape)

        for idx in test_idx:
            r, c = ratings[idx]
            train[r, c] = 0
            test[r, c] = matrix[r, c]

        pred = user_cf_k(train, k)
        rmse, _ = evaluate(pred, test)
        rmses.append(rmse)

    k_results.append(np.mean(rmses))

plt.figure()
plt.plot(k_values, k_results, marker='o')
plt.title("K vs RMSE (User-CF)")
plt.xlabel("K")
plt.ylabel("RMSE")
plt.grid()
plt.savefig("k_plot.png")

print("\nPlots saved: similarity_plot.png, k_plot.png")