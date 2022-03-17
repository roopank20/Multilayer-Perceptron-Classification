# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [Roopank Kohli] -- [rookohli]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    return sum(np.linalg.norm(val_x1 - val_x2) for val_x1, val_x2 in zip(x1, x2))

    # raise NotImplementedError('This function must be implemented by the student.')


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    return sum(abs(val_x1 - val_x2) for val_x1, val_x2 in zip(x1, x2))

    # raise NotImplementedError('This function must be implemented by the student.')


def identity(x, derivative=False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    if not derivative:
        return x
    else:
        return 1
    # raise NotImplementedError('This function must be implemented by the student.')


def sigmoid(x, derivative=False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    t = 1 / (1 + np.exp(-x))

    if not derivative:
        return t
    else:
        return t * (1 - t)

    # raise NotImplementedError('This function must be implemented by the student.')


def tanh(x, derivative=False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    t = np.tanh(x)

    if not derivative:
        return t
    else:
        return 1.0 - (t ** 2)

    # raise NotImplementedError('This function must be implemented by the student.')


def relu(x, derivative=False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    if not derivative:
        return np.where(x > 0, x, 0)
    else:
        return np.where(x > 0, 1, 0)

    # raise NotImplementedError('This function must be implemented by the student.')


def softmax(x, derivative=False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis=1, keepdims=True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis=1, keepdims=True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """
    y = np.clip(y, -1e100, 1e100)
    ce = -np.mean(p * np.log(y))
    return ce

    # raise NotImplementedError('This function must be implemented by the student.')


def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """

    nb_classes = len(set(y))
    targets = np.array([y]).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]

    return one_hot_targets

    # raise NotImplementedError('This function must be implemented by the student.')
