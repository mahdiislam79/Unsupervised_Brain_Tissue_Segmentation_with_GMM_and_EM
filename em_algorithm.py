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

def gaussian_pdf(data, mean, cov):
    """Calculate Gaussian probability density function with numerical stability."""
    size = len(data)
    cov += np.eye(size) * 1e-6  # Add a small value to the diagonal
    det = np.linalg.det(cov)
    norm_const = 1.0 / (np.power((2 * np.pi), float(size) / 2) * np.power(det, 1.0 / 2))
    data_diff = data - mean
    result = np.exp(-0.5 * np.sum(np.dot(data_diff, np.linalg.inv(cov)) * data_diff, axis=1))
    return norm_const * result


def em_algorithm(data, mu, sigma, pi, max_iter=100, tol=1e-6):
    n_samples, n_features = data.shape
    n_clusters = mu.shape[0]
    responsibilities = np.zeros((n_samples, n_clusters))
    log_likelihoods = []

    for iter in range(max_iter):
        # E-Step
        for i in range(n_clusters):
            responsibilities[:, i] = pi[i] * gaussian_pdf(data, mu[i], np.diag(sigma[i] ** 2))
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        # M-Step
        N_k = responsibilities.sum(axis=0)
        for i in range(n_clusters):
            mu[i] = (responsibilities[:, i].reshape(-1, 1) * data).sum(axis=0) / N_k[i]
            diff = data - mu[i]
            sigma[i] = np.sqrt((responsibilities[:, i].reshape(-1, 1) * diff ** 2).sum(axis=0) / N_k[i])
        pi = N_k / n_samples

        # Log-likelihood
        log_likelihood = np.sum(np.log(np.sum([pi[k] * gaussian_pdf(data, mu[k], np.diag(sigma[k] ** 2))
                                               for k in range(n_clusters)], axis=0)))
        log_likelihoods.append(log_likelihood)

        # Convergence check
        if iter > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return mu, sigma, pi, log_likelihoods, responsibilities
