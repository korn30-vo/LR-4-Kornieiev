import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.loadtxt("data_clustering.txt", delimiter=",")

kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    n_init=10,
    random_state=42
)

kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.05),
    np.arange(y_min, y_max, 0.05)
)

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")


plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)


plt.scatter(centers[:, 0], centers[:, 1],
            c="red", s=200, marker="X")

plt.title("K-Means Clustering (Full Method Version)")
plt.show()
