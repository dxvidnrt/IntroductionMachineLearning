import numpy as np

def prob_gaussian_multi_D(x_D, mean_D, cov_D_D):
    D = len(x_D)

    # Calculate the determinant and inverse of the covariance matrix
    det_cov_k = np.linalg.det(cov_D_D)
    inv_cov_k = np.linalg.inv(cov_D_D)

    # Compute the difference between x and the mean
    diff = x_D - mean_D  # Shape: (N, D)

    # Calculate the exponent term for the Gaussian formula
    exponent = -0.5 * np.sum(diff.T @ inv_cov_k @ diff)  # Shape: scalar

    # Normalization constant for Gaussian
    normalizer = (2 * np.pi) ** (D / 2) * np.sqrt(det_cov_k)

    # Gaussian probability for each component
    return np.exp(exponent) / normalizer