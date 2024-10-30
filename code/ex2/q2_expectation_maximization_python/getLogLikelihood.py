import numpy as np

from helpers import prob_gaussian_multi_D


def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar)

    #####Insert your code here for subtask 6a#####
    # Ln of all
    # Weight vector times probability vector for all gaussians
    # gaussian probability:
    N, D = X.shape
    K = len(means)

    logLikelihood = 0

    for x in range(N):

        # Initialize log likelihood
        likelihood_x = 0

        # Iterate over all Gaussian components
        for k in range(K):
            # Mean and covariance for Gaussian k
            mean_k = means[k]
            cov_k = covariances[:, :, k]
            x_D = X[x, :]

            # Gaussian probability for each component
            prob_k = prob_gaussian_multi_D(x_D, mean_k, cov_k)

            # Weighted probability
            weighted_prob_k = weights[k] * prob_k  # Shape: (N,)

            # Accumulate the log likelihood
            likelihood_x += np.sum(weighted_prob_k)

        log_likelihood_x = np.log(likelihood_x)

        logLikelihood += log_likelihood_x

    return logLikelihood

