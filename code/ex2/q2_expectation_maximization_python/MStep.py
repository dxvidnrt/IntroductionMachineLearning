import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).
    N, K = gamma.shape
    _, D = X.shape

    means = np.zeros((K, D))
    weights = np.zeros(K)
    covariances = np.zeros((D, D, K))

    N_est = np.zeros(K)  # 1D array for expected counts
    for j in range(K):
        N_j = np.sum(gamma[:, j])  # Sum over all data points for the j-th Gaussian
        N_est[j] = N_j

    for j in range(K):
        N_j = N_est[j]
        mu_j = np.zeros(D)
        for x in range(N):
            mu_j += gamma[x, j] * X[x]
        mu_j /= N_j
        means[j] = mu_j
        weights[j] = N_j / N

    for j in range(K):
        N_j = N_est[j]
        cov_sum = np.zeros((D, D))
        for x in range(N):
            diff = X[x] - means[j]
            cov_sum += gamma[x, j] * np.outer(diff, diff)

        covariances[:, :, j] = cov_sum / N_j

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    #####Insert your code here for subtask 6c#####
    return weights, means, covariances, logLikelihood
