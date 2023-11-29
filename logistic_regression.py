import numpy as np

"""
    This module implements logistic regression with a ridge pentaly without the intercept functionality.
"""

def log_odds(B, X):
    z = np.sum(B * X, axis=1)
    p = 1 / (1 + np.exp(-(z)))
    return p


def predict(X, B):
    # ones = np.ones_like(X[:, 0])
    # ones = np.expand_dims(ones, axis=1)  # Reshape ones to have 2 dimensions
    # X = np.hstack((ones, X))

    p = log_odds(B, X)

    res = np.where(p >= 0.5, 1, 0)
    return res


def ridge_gradient(X, c, B, lam):
    # ones = np.ones_like(X[:, 0])
    # ones = np.expand_dims(ones, axis=1)  # Reshape ones to have 2 dimensions
    # X = np.hstack((ones, X))
    
    p = log_odds(B, X)

    t0 = (c * (1 - p))[:, np.newaxis] * X
    t1 = ((1 - c) * p)[:, np.newaxis] * X
    gradient = -1 * np.sum(t0 , axis=0) + np.sum(t1, axis=0)

    ridge_term = np.squeeze(2 * lam * B)
    gradient += ridge_term

    return gradient


def stopping_condition(B, B_new, threshold=0.01):
    return np.sum(B - B_new) < threshold


def fit(X, y, lam, eta=0.01, decay_factor=0.9, max_iter=1e4):
    # B = np.zeros(len(X[0])+1)
    B = np.zeros(len(X[0]))
    
    i = 0
    while i < 1e4:
        gradient = ridge_gradient(X, y, B, lam)
        B_new = -eta * gradient + B
        eta = eta * decay_factor

        if stopping_condition(B, B_new):
            return B_new

        B = B_new
        i += 1

    return B