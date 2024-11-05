from perceptron import Perceptron

if __name__ == '__main__':
    # 1 refers to class 2, 0 refers to class 1
    data = [[-0.5, 3.5, 0],
                   [0.5, 3.5, 0],
                   [1, 3, 0],
                   [-0.5, 2.5, 0],
                   [0.5, 2.5, 0],
                   [1, 2, 0],
                   [-1, 1.5, 0],
                   [-0.5, 1, 0],
                   [1, 1, 0],
                   [0.5, 0.5, 0],
                   [2, 0.5, 0],
                   [-0.5, -0.5, 0],
                   [0.5, -0.5, 0],
                   [1.5, -0.5, 0],
                   [2.5, -0.5, 0],
                   [-0.5, -1.5, 0],
                   [1, -1.5, 0],
                   [5, 5, 1],
                   [6.5, 4.5, 1],
                   [5.5, 4, 1],
                   [5, 3.5, 1],
                   [7, 3.5, 1],
                   [6, 3, 1],
                   [7.5, 2.5, 1],
                   [6.5, 2, 1],
                   [7.5, 1.5, 1],
                   [8.5, 1.5, 1],
                   [8, 0.5, 1],
                   [7.5, -0.5, 1],
                   [7, -1, 1],
                   [8.5, -1, 1],
                   [7.5, -1.5, 1]]

    learning_rate = 0.9
    number_of_epochs = 100

    perceptron = Perceptron(data, learning_rate, number_of_epochs)
    perceptron.initialize_weights()
    perceptron.adjust_weights()
    prediction = perceptron.predict([0.5, 2.5])
    print(f"Prediction: {prediction}")