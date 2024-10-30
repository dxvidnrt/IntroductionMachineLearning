import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####

    def prob_gaussian(x, mu, variance):
        normalizer = 1 / (np.sqrt(2 * np.pi) * (variance)**2)
        exponent = np.exp(-1 * ((x - mu)**2) / (2 * (variance ** 2)))
        return normalizer * exponent

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    N, _ = X.shape
    K = len(weights)
    gamma = np.zeros(N, K)
    variances = np.array([np.diagonal(covariances[:, :, k]) for k in range(K)])

    for x in range(N):
        sum_posterior_gaussians = 0
        for k in range(K):
            prior_k = weights[k]
            prob_k = prob_gaussian(x, means[k], variances[k])
            sum_posterior_gaussians += prior_k * prob_k

        for j in range(K):
            # j is current Gaussian
            prior_j = weights[j]
            prob_j = prob_gaussian(x, means[j], variances[j])
            y_j_x_n = (prior_j, prob_j) / sum_posterior_gaussians
            gamma[x][j] = y_j_x_n

    return [logLikelihood, gamma]
