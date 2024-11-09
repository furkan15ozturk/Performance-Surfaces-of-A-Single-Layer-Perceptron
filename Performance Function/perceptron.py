from random import random

import numpy as np


class Perceptron:
    def __init__(self, data, learning_rate, alpha, number_of_epochs, F, method):
        # Initialize the perceptron with the given data, learning rate, alpha, number of epochs, F, and method
        self.data = data
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.number_of_epochs = number_of_epochs
        self.F = F
        self.method = method

        # Modify the given data to training_data
        self.training_data = data
        # Get the number of features from the data
        self.number_of_features = data.shape[1]
        # Initialize the weights list, weights_over_time list, and d_q_n_values
        self.weights = []
        self.weights_over_time = []
        self.d_q_n_values = []

    def get_weights_over_time(self):
        return self.weights_over_time

    def initialize_weights(self):
        for i in range(self.number_of_features):
            #In the paper, the weights are initialized to 0
            self.weights.append(0)

    # This method calculates the dot product of the F vector and the Input vector
    def calculated_d(self, f_vector, x_n):
        return np.dot(f_vector, x_n)

    # This method calculates the d_q_n values for the training data and assigns each d_q_n value to d_q_n_values
    def calculate_d_q_n(self):
        self.d_q_n_values = [1 if np.dot(self.F, x_n) > 0 else -1 for x_n in self.training_data]
        return self.d_q_n_values

    def calculate_y_q_n(self, dot_product):
        return self.sign(dot_product)

    # f(y(n)) function that is in the performance function B
    def f(self, y_n):
        return 1 / (1 + np.exp(-self.alpha * y_n))

    @staticmethod
    def sign(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def weighted_sum(self, x_n):
        return np.dot(self.weights, x_n)

    def fit(self):
        self.initialize_weights()
        self.d_q_n_values = self.calculate_d_q_n()

        print(f"  Initial Weights: {self.weights}")
        print("-" * 40)

        for epoch in range(self.number_of_epochs):
            for i, x_n in enumerate(self.training_data):
                y_n = self.weighted_sum(x_n)
                y_q_n = self.calculate_y_q_n(y_n)
                d_q_n = self.d_q_n_values[i]
                e_q_n = d_q_n - y_q_n
                for j in range(self.number_of_features):
                    if self.method == 1:
                        # W(n+1) = W(n) + 2*learning_rate*(d(n) - y(n))*X(n)
                        self.weights[j] += 2 * self.learning_rate * e_q_n * self.training_data[i][j]
                    elif self.method == 2:
                        # W(n+1) = W(n) + 4*learning_rate*alpha*e_q(n)*f(y(n))*(1-f(y(n)))*X(n)
                        self.weights[j] += 4 * self.learning_rate * self.alpha * e_q_n * self.f(y_n) * (1 - self.f(y_n)) * self.training_data[i][j]
                    elif self.method == 3:
                        # W(n+1) = W(n) + 2*learning_rate*(d_q(n) - y(n))*X(n)
                        self.weights[j] += 2 * self.learning_rate * (d_q_n - y_n) * self.training_data[i][j]
                self.weights_over_time.append(self.weights.copy())

    def predict(self, user_input):
        dot_product = 0
        for i in range(len(user_input)):
            dot_product += user_input[i] * self.weights[i]
        return dot_product, 1 if dot_product >= 0 else -1


    def print_var(self):
        print(f"Data: {self.data}")
        print(f"Training data: {self.training_data}")
        print(f"Number of features: {self.number_of_features}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Number of epochs: {self.number_of_epochs}")
        print(f"Weights: {self.weights}")
