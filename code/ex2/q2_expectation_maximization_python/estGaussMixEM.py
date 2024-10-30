import numpy as np
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def estGaussMixEM(data, K, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : NxD input data. N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : K mixture weights - P(j) from lecture
    # means          : KxD means of gaussians
    # covariances    : DxDxK covariance matrices of gaussians

    n_dim = data.shape[1]

    # initialize weights and covariances
    weights = np.ones(K) / K
    covariances = np.zeros((n_dim, n_dim, K))

    # Use k-means for initializing the EM-Algorithm.
    # cluster_idx: cluster indices
    # means: cluster centers
    kmeans = KMeans(n_clusters=K, n_init=10).fit(data)
    cluster_idx = kmeans.labels_
    means = kmeans.cluster_centers_

    # Create initial covariance matrices
    for j in range(K):
        data_cluster = data[cluster_idx == j]
        min_dist = np.inf
        for i in range(K):
            # compute sum of distances in cluster
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
            if dist < min_dist:
                min_dist = dist
        covariances[:, :, j] = np.eye(n_dim) * min_dist


    #####Insert your code here for subtask 6e#####
    return [weights, means, covariances]
