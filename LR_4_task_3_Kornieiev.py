import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

X = np.loadtxt("data_clustering.txt", delimiter=",")
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_
centers = ms.cluster_centers_
n_clusters = len(np.unique(labels))

print("Number of clusters:", n_clusters)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
plt.scatter(centers[:, 0], centers[:, 1],
            c="red", s=200, marker="X")

plt.title("Mean Shift Clustering")
plt.show()
