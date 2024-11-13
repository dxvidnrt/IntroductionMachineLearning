import numpy as np

# Initialize matrix X and vector t
X = np.array([
    [1, 1],
    [1, 4],
    [1, 7],
    [1, 10],
    [1, 6]
])

t = np.array([2, 5, 8, 5, 3])


# Define a function to calculate w using the normal equation
def calculate_weights(X, t):
    # Convert t to a column vector
    t = t.reshape(-1, 1)

    # Calculate w = (X^T X)^-1 X^T t
    X_transpose = X.T
    w = np.linalg.inv(X_transpose @ X) @ X_transpose @ t
    return w


# Compute w
w = calculate_weights(X, t)

# Print result
print("Calculated weights (w):")
print(w)
