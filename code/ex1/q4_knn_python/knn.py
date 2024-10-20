import numpy as np


def knn(samples, k):
    """
    compute density estimation from samples with k-NN
    Input
     samples    : (N,) vector of data points
     k          : number of neighbors
    Output
     estimatedDensity : (200, 2) estimated density in the range of [-5, 5]
    """

    #####Insert your code here for subtask 5b#####
    N = len(samples)
    sorted_samples = np.sort(samples)

    pos = np.arange(-5.0, 5.0, 0.05)

    def kth_nearest_neighbour_distance(x, k_neighbours):
        differences = np.abs(sorted_samples - x)
        return np.partition(differences, k_neighbours - 1)[k_neighbours - 1]

    estimatedDensity = []
    for x in pos:
        v_k = kth_nearest_neighbour_distance(x, k)
        p_x = k / (N * v_k)
        estimatedDensity.append(p_x)


    estimatedDensity = np.stack((pos, estimatedDensity), axis=1)

    return estimatedDensity
