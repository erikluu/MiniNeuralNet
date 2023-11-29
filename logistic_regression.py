import numpy as np


def log_odds(X, B):
    z = np.sum(B * X, axis=1)
    p = 1 / (1 + np.exp(-(z)))
    return p


def predict(X, B):
    """
    Predict the class labels using logistic regression.
    
    Parameters:
    X (numpy.ndarray): The input features.
    B (numpy.ndarray): The coefficients of the logistic regression model.
    
    Returns:
    numpy.ndarray: The predicted class labels.
    """
    ones = np.ones_like(X[:, 0])
    ones = np.expand_dims(ones, axis=1)  # Reshape ones to have 2 dimensions
    X = np.hstack((ones, X))

    p = log_odds(B, X)

    res = np.where(p >= 0.5, 1, 0)
    return res


def ridge_gradient(X, c, B, lam):
    ones = np.ones_like(X[:, 0])
    ones = np.expand_dims(ones, axis=1)  # Reshape ones to have 2 dimensions
    X = np.hstack((ones, X))
    
    p = log_odds(B, X)

    t0 = (c * (1 - p))[:, np.newaxis] * X
    t1 = ((1 - c) * p)[:, np.newaxis] * X
    gradient = -1 * np.sum(t0 , axis=0) + np.sum(t1, axis=0)

    ridge_term = np.squeeze(2 * lam * B)
    gradient += ridge_term

    return gradient


def stopping_condition(B, B_new, threshold=0.01):
    return np.sum(B - B_new) < threshold


def fit(X, c, lam=10, eta=.01, decay_factor=1, max_iter=1e4):
    """
    Fit a logistic regression model with ridge regularization.
    
    Parameters:
    X (numpy.ndarray): The input features.
    c (numpy.ndarray): The true class labels.
    lam (float): The regularization parameter (lambda).
    eta (float): The learning rate.
    decay_factor (float): The decay factor for the learning rate.
    max_iter (int): The maximum number of iterations.
    
    Returns:
    numpy.ndarray: The coefficients of the logistic regression model.
    """
    B = np.zeros(len(X[0])+1)
    
    for _ in range(max_iter):
        gradient = ridge_gradient(X, c, B, lam)
        B_new = -eta * gradient + B
        eta = eta * decay_factor

        if stopping_condition(B, B_new):
            return B_new

        B = B_new

    return B