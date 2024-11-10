import numpy as np
from sklearn.cluster import KMeans

def initialize_clusters(data, n_clusters=3, random_state=0):
    """Initialize clusters using k-means for mu, sigma, and pi."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data)
    mu = kmeans.cluster_centers_
    labels = kmeans.labels_
    sigma = np.array([np.std(data[labels == i], axis=0) for i in range(n_clusters)])
    pi = np.ones(n_clusters) / n_clusters  # Equal probability for each cluster initially
    return mu, sigma, pi

def gaussian_pdf(data, mean, cov):
    """Calculate Gaussian probability density function with numerical stability."""
    size = len(data)
    cov += np.eye(size) * 1e-6  # Add small value to diagonal for stability
    det = np.linalg.det(cov)
    norm_const = 1.0 / (np.power((2 * np.pi), float(size) / 2) * np.power(det, 1.0 / 2))
    data_diff = data - mean
    result = np.exp(-0.5 * np.sum(np.dot(data_diff, np.linalg.inv(cov)) * data_diff, axis=1))
    return norm_const * result

def em_algorithm(data, max_iter=100, tol=1e-6, n_clusters=3):
    """EM algorithm for Gaussian Mixture Models with k-means initialization."""
    # Initialize using KMeans
    mu, sigma, pi = initialize_clusters(data, n_clusters)

    n_samples, n_features = data.shape
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

        # Log-likelihood calculation
        log_likelihood = np.sum(np.log(np.sum([pi[k] * gaussian_pdf(data, mu[k], np.diag(sigma[k] ** 2))
                                               for k in range(n_clusters)], axis=0)))
        log_likelihoods.append(log_likelihood)

        # Convergence check
        if iter > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return mu, sigma, pi, log_likelihoods, responsibilities
