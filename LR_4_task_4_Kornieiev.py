import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
import yfinance as yf

symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]

data = yf.download(symbols, start="2024-01-01", end="2025-01-01")["Close"]

returns = data.pct_change().dropna().T.values

scaler = StandardScaler()
X = scaler.fit_transform(returns)

model = AffinityPropagation(random_state=42)
model.fit(X)

labels = model.labels_
exemplars = model.cluster_centers_indices_

print("Clusters:", len(np.unique(labels)))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50)

plt.title("Affinity Propagation Clustering")
plt.show()
