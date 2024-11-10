import numpy as np
from sklearn.cluster import KMeans

def initialize_clusters(data, n_clusters=3, random_state=0):
    """Initialize clusters using k-means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data)
    return kmeans.cluster_centers_, kmeans.labels_

def calculate_sigma(data, labels, n_clusters=3):
    """Calculate standard deviation for each cluster."""
    sigma = []
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        cluster_std = np.std(cluster_points, axis=0)
        sigma.append(cluster_std)
    return np.array(sigma)

def em_algorithm(data, mu, sigma, pi, max_iters=100, tol=1e-4):
    """Expectation-Maximization algorithm for clustering."""
    n, d = data.shape
    k = len(mu)
    responsibilities = np.zeros((n, k))
    
    for iteration in range(max_iters):
        # E-step
        for i in range(k):
            responsibilities[:, i] = pi[i] * gaussian_pdf(data, mu[i], sigma[i])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        # M-step
        N_k = responsibilities.sum(axis=0)
        mu = (responsibilities.T @ data) / N_k[:, None]
        sigma = np.array([
            np.sqrt((responsibilities[:, i] * ((data - mu[i]) ** 2)).sum(axis=0) / N_k[i])
            for i in range(k)
        ])
        pi = N_k / n
        
        # Check for convergence
        if np.allclose(mu, mu, atol=tol):
            break
    
    return mu, sigma, pi, responsibilities

def gaussian_pdf(data, mu, sigma):
    """Calculate the Gaussian probability density function for given data."""
    return np.exp(-0.5 * ((data - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
