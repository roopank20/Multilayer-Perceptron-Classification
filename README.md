# Multilayer-Perceptron-Classification

## Problem Description

Multi-layer perceptron is a type of network where multiple layers of a group of perceptron are stacked together to make a model. In this case,  perceptron/neuron as a linear model which takes multiple inputs and produce an output. In this case perceptron takes a bunch of inputs multiply them with weights and add a bias term to generate an output. The objective in this part is to implement a feedforward fully-connected multilayer perceptron classifier with one hidden layer, from scratch. Along with this, certain activation functions were needed to be implemented in the utils.py file.

# Solution and Approach

Similar to the previous problem, here I was provided with two different datasets namely - Iris dataset and Digits dataset, having distinct set of features associated with them. 

The code is implemented in two parts. First part is to train the classifier with the help of training dataset and second one is to test the classifier by predicting the results using the testing data.

Steps to train the classifier are mentioned below.

* Define and initialise the attributes like activation function, hidden layer perceptron count, number of iterations, learning rate, loss function, output activation function, weights for hidden and output layer, bias value for hidden and output layer.

* Random values were assigned to the weights and bias and one-hot encoding was done for the array containing the class values.

* Propagate all values in the input layer until output layer(Forward Propagation). This propagation was done with the help of activation functions. The activation function does the non-linear transformation to the input making it capable to learn and perform more complex tasks.

* Update weight and bias in the inner layers(Backpropagation) using the derivatives of previously defined values.

After training the neural network based classifier, the next step was to test the classifier on some testing data points.

## Design Decisions

A number of data structures including list, array, set etc. and a variety of inbuilt functions like np.dot() (for matrix) multiplication), np.sort(), max(), np.argsort(), np.random() along with list comprehensions were used in this solution. The main approach to decide the designing decisions for this problem was to implement the solution in minimum time complexity and provide maximum accuracy. The output here displays the accuracy attained by our classifier on this 
specific dataset. 

## Assumptions

It was assumed that the input data features are all numerical features and that the target class values are categorical features.

## Problems Faced
* It was found that the running time for digits dataset was comparatively longer than the iris dataset.
* Complicated matrix multiplication along with derivation computation was difficult to perform.
* Computation time was too high without using the numpy library.
