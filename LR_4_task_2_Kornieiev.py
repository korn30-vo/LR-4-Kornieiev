import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    random_state=42
)

labels = kmeans.fit_predict(X)

centers = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=labels, cmap="viridis", s=40)

plt.scatter(centers[:, 0], centers[:, 1],
            c="red", s=200, marker="X")

plt.title("Iris Dataset Clustering (K-Means + PCA)")
plt.show()
