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

    and_input = [[0, 0, 0],
                 [0, 1, 0],
                 [1, 0, 0],
                 [1, 1, 1]]

    learning_rate = 0.01
    number_of_epochs = 1000

    perceptron = Perceptron(and_input, learning_rate, number_of_epochs)
    perceptron.initialize_weights()
    perceptron.fit()
    prediction = perceptron.predict([4.2, 5])
    weights_over_time = perceptron.get_weights_over_time()
    performance_over_time = perceptron.get_performance_over_time()
    print(f"Prediction: {prediction}")
    # for i in range(len(performance_over_time)):
    #     print(f"Performance over time: {performance_over_time[i]} | Weights over time: {weights_over_time[i]}")