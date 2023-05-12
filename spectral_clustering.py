import numpy as np
from sklearn.cluster import KMeans

class SpectralClustering:
    def __init__(self, n_clusters, sigma):
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.means = None

    def fit(self, X):
        pairwise_dists_sq = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1)[np.newaxis, :] - 2*np.dot(X, X.T)
        W = np.exp(-pairwise_dists_sq / (2 * self.sigma**2))

        # Compute the degree matrix
        D = np.diag(np.sum(W, axis=1))

        # Compute the unnormalized Laplacian matrix
        L = D - W

        # Compute the first k eigenvectors of L
        eigvals, eigvecs = np.linalg.eig(L)
        indices = np.argsort(eigvals)[:self.n_clusters]
        V = eigvecs[:, indices].real.astype(np.double)

        # Cluster the rows of V using k-means
        self.means = KMeans(n_clusters=self.n_clusters)
        self.means.fit(V)
        

    def predict(self, X):
        return self.means.predict(X)


# Run spectral clustering

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=3, n_features=2)
n_clusters = 3
sigma = 0.1

SC = SpectralClustering(2, 0.1)
SC.fit(X)
labels = SC.predict(X)
print(X)

# Visualize the results
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()