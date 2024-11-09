from perceptron import Perceptron
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Reading CSV Data
    df = pd.read_csv('../Data Generation/jointly_gaussian_data.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    data = df.values

    #Initializing necessary variables
    learning_rate = 0.01
    alpha = 1
    #Since the data has 1000 data points, the number of epochs is set to 1 in order to keep the iteration number 1000.
    number_of_epochs = 1
    # F can be [-1, 1] or [-1, -1]
    F = [-1, 1]

    # Choosing the method.
    # Method = 1 for the performance function in A
    # Method = 2 for the performance function in B
    # Method = 3 for the performance function in C

    method = 2

    #Initializing the perceptron


    #Weights over time
    runs = 25
    independent_weight_trajectories = []
    mean_trajectory = []
    for i in range (runs):
        #In the "Examples of the Performance Surfaces" section, it is said that 25 computer runs are completed independently.
        perceptron = Perceptron(data, learning_rate, alpha, number_of_epochs, F, method)
        perceptron.fit()
        independent_weight_trajectories.append(perceptron.get_weights_over_time())

    independent_weight_trajectories = np.array(independent_weight_trajectories)
    mean_trajectory = np.mean(independent_weight_trajectories, axis=0)
    mean_trajectory = np.insert(mean_trajectory, 0, [0, 0], axis=0)

    print("Mean trajectory shape:", mean_trajectory.shape)  # Should be (1000, 2)
    print("Mean trajectory for first few iterations:", mean_trajectory[:15])

    mean_df = pd.DataFrame(mean_trajectory, columns=['mean_w1', 'mean_w2'])

    mean_df.to_csv(f'weight_trajectories_performance_{method}.csv', index_label='iteration')