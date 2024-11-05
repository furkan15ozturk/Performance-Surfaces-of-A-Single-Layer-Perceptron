import random

class Perceptron:
    def __init__(self, data, learning_rate, number_of_epochs):
        self.data = data
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs

        self.training_data = [[1] + row[:-1] for row in self.data]
        self.number_of_features = len(self.training_data[0])
        self.output_data = [row[-1] for row in self.data]
        self.weights = []
        self.weights_over_time = []
        self.performance_over_time = []
        self.performance = 0

    def get_weights_over_time(self):
        return self.weights_over_time

    def get_performance_over_time(self):
        return self.performance_over_time

    def initialize_weights(self):
        for i in range(self.number_of_features):
            self.weights.append(random.random())
            self.weights[0] = 0

    def calculated_desired_response(self, output):
        return 1 if output == 1 else -1

    @staticmethod
    def dot_product(vector1, vector2):
        # Ensure both vectors are of the same length
        if len(vector1) != len(vector2):
            raise ValueError("Vectors must be of the same length.")

        # Calculate the dot product
        result = sum(a * b for a, b in zip(vector1, vector2))
        return result

    def calculate_y(self, dot_product):
        return self.sign(dot_product)
    # Returns quantized output

    @staticmethod
    def sign(x):
        return 1 if x >= 0 else -1

    def weighted_sum(self, row):
        dot_product = 0
        for i in range(self.number_of_features):
            dot_product += row[i] * self.weights[i]
        return dot_product

    def fit(self):
        for _ in range(self.number_of_epochs):
            self.weights_over_time.append(self.weights.copy())
            self.performance_over_time.append(self.performance)
            for i in range(len(self.training_data)):
                y_n = self.weighted_sum(self.training_data[i])
                y_q_n = self.calculate_y(y_n)
                d_q_n = self.calculated_desired_response(self.output_data[i])
                for j in range(self.number_of_features):
                    # W(n+1) = W(n) + 2*learning_rate*(d(n) - y(n))*x(n)
                    self.weights[j] += 2*self.learning_rate * (d_q_n - y_q_n) * self.training_data[i][j]
                self.performance = self.performance_function(y_n, d_q_n)

    def predict(self, user_input):
        dot_product = 0
        user_input = [1] + user_input
        for i in range(len(user_input)):
            dot_product += user_input[i] * self.weights[i]
        prediction = self.sign(dot_product)
        if prediction == 1:
            return 1
        elif prediction == -1:
            return 0

    def performance_function(self, y_n, d_q_n):
        perf = 2*abs(y_n) - 2 * d_q_n * y_n
        # print(2*abs(y_n))
        # print(2 * d_q_n * y_n)
        return perf

    def print_var(self):
        print(f"Data: {self.data}")
        print(f"Training data: {self.training_data}")
        print(f"Number of features: {self.number_of_features}")
        print(f"Output data: {self.output_data}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Number of epochs: {self.number_of_epochs}")
        print(f"Weights: {self.weights}")
        for row in self.training_data:
            dot_product = self.dot_product(row)  # dot product of  ith row
            print(f"Row {row}: Dot product: {dot_product}")
