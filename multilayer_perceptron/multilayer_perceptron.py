# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [Roopank Kohli] -- [rookohli]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

#### REFERENCES ####

# This is to declare that this problem was discussed with Anurag Hambir and Akash Bhapkar. I certify that our discussion
# was limited to logical discussion and theoritical reasoning. No discussion related to implementation and coding took
# place.

# The below mentioned resources were referred to implement the solution for this specific problem.

# This videos was referred to understand the intuition and explanation of the algorithm.
# https://www.youtube.com/watch?v=u5GAVdLQyIg
# https://www.youtube.com/watch?v=IlmNhFxre0w

# This link was referred to gather theoritical logic and explanation for the multilayer perceptron classifier
# https://medium.com/engineer-quant/multilayer-perceptron-4453615c4337

# This link was referred to gain knowledge about activation functions, its graphical explanation, its utility and its
# influence over the classification process
# https://cup-of-char.com/writing-activation-functions-from-mostly-scratch-in-python/




import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding
import math


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden=16, hidden_activation='sigmoid', n_iterations=1000, learning_rate=0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._X = X
        self._y = one_hot_encoding(y)

        N = self._X.shape[1]
        M = self._y.shape[1]

        a = 1 / math.sqrt(N)
        b = 1 / math.sqrt(M)

        # Initialising weights and bias values for both hidden layer and output layer. The dimensions were taken as per
        # instructions provided in the description.
        self._h_weights = np.random.uniform(-a, a, (N, self.n_hidden))
        self._h_bias = np.random.uniform(-a, a, (1, self.n_hidden))
        self._o_weights = np.random.uniform(-b, b, (self.n_hidden, M))
        self._o_bias = np.random.uniform(-b, b, (1, M))

        np.random.seed(42)

        # raise NotImplementedError('This function must be implemented by the student.')

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
        iterations.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._initialize(X, y)

        # Training the classifier with the dataset.
        for _ in range(self.n_iterations):
            # FORWARD PROPAGATION

            zHidden = np.dot(self._X, self._h_weights) + self._h_bias
            aHidden = self.hidden_activation(zHidden)

            zOutput = np.dot(aHidden, self._o_weights) + self._o_bias
            aOutput = self._output_activation(zOutput)

            # BACKWARD PROPAGATION

            derivative_zOutput = aOutput - self._y
            derivativeWeightOutput = np.dot(aHidden.T, derivative_zOutput)
            derivativeBiasOutput = (1 / self._X.shape[0]) * np.sum(derivative_zOutput, axis=0, keepdims=True)

            derivative_zHidden = np.dot(derivative_zOutput, self._o_weights.T) * self.hidden_activation(zHidden, True)
            derivativeWeightHidden = (1 / self._X.shape[0]) * np.dot(self._X.T, derivative_zHidden)
            derivativeBiasHidden = (1 / self._X.shape[0]) * np.sum(derivative_zHidden, axis=0, keepdims=True)

            self._h_weights = self._h_weights - self.learning_rate * derivativeWeightHidden
            self._o_weights = self._o_weights - self.learning_rate * derivativeWeightOutput
            self._h_bias = self._h_bias - self.learning_rate * derivativeBiasHidden
            self._o_bias = self._o_bias - self.learning_rate * derivativeBiasOutput

        # raise NotImplementedError('This function must be implemented by the student.')

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        zHiddenTest = np.dot(X, self._h_weights) + self._h_bias
        aHiddenTest = self.hidden_activation(zHiddenTest)
        zOutputTest = np.dot(aHiddenTest, self._o_weights) + self._o_bias
        aOutputTest = self._output_activation(zOutputTest)

        outputList = []

        for row in aOutputTest:
            op = np.argmax(row, axis=0)
            outputList.append(op)

        output = np.array(outputList)

        return output

        # raise NotImplementedError('This function must be implemented by the student.')
