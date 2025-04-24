from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from cProfile import label


# Step 1: Generate the dataset
np.random.seed(42)

cluster1 = np.random.multivariate_normal([1, 2], [[0.1, 0.05], [0.05, 0.2]], 1000)
cluster2 = np.random.multivariate_normal([0, 0], [[0.3, -0.1], [-0.1, 0.2]], 500)
cluster3 = np.random.multivariate_normal([-2, 3], [[1.5, 0], [0, 1.5]], 1500)

data = np.vstack((cluster1, cluster2, cluster3))
labels_true = np.array([0] * 1000 + [1] * 500 + [2] * 1500)

# Step 2: Tune DBSCAN hyperparameters
eps_values = np.arange(0.1, 1.1, 0.1)
min_samples_values = np.arange(1, 11, 1)

best_ari = -1
best_eps = None
best_min_samples = None
best_labels = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        
        # Ignore cases where all points are outliers (-1 labels only)
        if len(set(labels)) > 1:
            ari = adjusted_rand_score(labels_true, labels)
            if ari > best_ari:
                best_ari = ari
                best_eps = eps
                best_min_samples = min_samples
                best_labels = labels

# Step 3: Print best parameters and ARI
print(f"Best eps: {best_eps}")
print(f"Best min_samples: {best_min_samples}")
print(f"Best Adjusted Rand Index: {best_ari}")

# Step 4: Plot results
plt.figure(figsize=(8, 6))
unique_labels = set(best_labels)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    mask = (best_labels == label)
    color = 'k' if label == -1 else colors(label)
    plt.scatter(data[mask, 0], data[mask, 1], label=f"Cluster {label}" if label != -1 else "Outliers", alpha=0.6)

plt.legend()
plt.title(f"DBSCAN Clustering (eps={best_eps}, min_samples={best_min_samples})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
