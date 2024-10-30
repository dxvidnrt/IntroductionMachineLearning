import numpy as np


def regularize_cov(covariance, epsilon):
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values. Explanation see exercise sheet.
    #
    # INPUT:
    # covariance     : DxD matrix
    # epsilon        :    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix

    #####Insert your code here for subtask 6d#####
    D = covariance.shape[0]
    regularized_cov = covariance + epsilon * np.eye(D)

    return regularized_cov
