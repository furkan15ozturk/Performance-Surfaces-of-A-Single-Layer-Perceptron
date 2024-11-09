import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 2  # Number of signals (dimensions)
M = 1000 # Number of samples
mean = 0
R = np.identity(N)

# Generate M samples of N-dimensional jointly Gaussian signals
data = np.random.multivariate_normal([mean]*N, R, M)
# data now contains M samples of two jointly Gaussian signals

mean_vector_before = np.mean(data, axis=0)
print("Calculated Mean of the Dataset:", mean_vector_before)

print("Preview of the Dataset Before:")
print(data[:5])

data -= mean_vector_before

mean_vector_after = np.mean(data, axis=0)
print("Calculated Mean of the Dataset after mean subtraction:", mean_vector_after)

print("Preview of the Dataset After:")
print(data[:5])

np.savetxt('jointly_gaussian_data.csv', data, delimiter=',', header='Signal 1,Signal 2', comments='')

print("Data has been written to 'jointly_gaussian_data.csv'")
