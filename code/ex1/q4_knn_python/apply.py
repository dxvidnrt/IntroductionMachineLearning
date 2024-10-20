import matplotlib.pyplot as plt
import numpy as np
from gauss1D import gauss1D
from kde import kde
from knn import knn
from parameters import parameters

h, k = parameters()

print("Question: Kernel/K-Nearest Neighborhood Density Estimators")

# Produce the random samples
samples = np.random.normal(0, 1, 100)

# Compute the original normal distribution
realDensity = gauss1D(0, 1, 200, 5)

# Estimate the probability density using the KDE
estimatedDensity = kde(samples, h)

# plot results
plt.subplot(2, 1, 1)
plt.plot(
    estimatedDensity[:, 0],
    estimatedDensity[:, 1],
    "r",
    linewidth=1.5,
    label="KDE Estimated Distribution",
)
plt.plot(
    realDensity[:, 0], realDensity[:, 1], "b", linewidth=1.5, label="Real Distribution"
)
plt.legend()

# Estimate the probability density using KNN
estimatedDensity = knn(samples, k)

# Plot the distributions
plt.subplot(2, 1, 2)
plt.plot(
    estimatedDensity[:, 0],
    estimatedDensity[:, 1],
    "r",
    linewidth=1.5,
    label="KNN Estimated Distribution",
)
plt.plot(
    realDensity[:, 0], realDensity[:, 1], "b", linewidth=1.5, label="Real Distribution"
)
plt.legend()
plt.show()