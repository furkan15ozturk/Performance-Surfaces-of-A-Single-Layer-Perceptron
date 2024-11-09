import numpy as np
import matplotlib.pyplot as plt

# Define functions based on given formulas
def f(y, alpha):
    return 1 / (1 + np.exp(-alpha * y))

def g(y, alpha):
    return 2 * f(y, alpha) - 1

def g_prime(y, alpha):
    f_y = f(y, alpha)
    return 2 * alpha * f_y * (1 - f_y)

# Define y values for plotting
y_values = np.linspace(-4, 4, 400)

# List of alpha values to plot
alpha_values = [1, 2, 10]

# Plot for g(y(n)) = 2f(y(n)) - 1
plt.figure(figsize=(6, 6))
for alpha in alpha_values:
    g_values = g(y_values, alpha)
    plt.plot(y_values, g_values, label=f'α = {alpha}')
plt.xlabel('Nonlinearity Input y')
plt.ylabel('Nonlinearity Output g(y)')
plt.title('Sigmoidal nonlinearity for different α values (g(y))')
plt.legend()
plt.grid(True)
plt.savefig('sigmoidal_nonlinearity_g.svg', format='svg', bbox_inches='tight')
plt.show()

# Plot for g'(y(n)) = 2α f(y(n)) [1 - f(y(n))]
plt.figure(figsize=(6, 6))
for alpha in alpha_values:
    g_prime_values = g_prime(y_values, alpha)
    plt.plot(y_values, g_prime_values, label=f'α = {alpha}')
plt.xlabel('Nonlinearity Input y')
plt.ylabel('Nonlinearity Output g\'(y)')
plt.title('Sigmoidal nonlinearity for different α values (g\'(y))')
plt.legend()
plt.grid(True)
plt.savefig('sigmoidal_nonlinearity_g_prime.svg', format='svg', bbox_inches='tight')
plt.show()
