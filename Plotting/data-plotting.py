import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV data
df = pd.read_csv('../Data Generation/jointly_gaussian_data.csv')

F = [-1, 1]
weights=[-2.2532756068842543, 2.25801753257601]
x_vals = np.array([-15, 15])  # X aralığı ayarlanabilir

# x2'nin değerini hesaplayın: x2 = -(f1 / f2) * x1
slope = -F[0] / F[1]  # Eğimi belirleyin
y_vals = slope * x_vals  # y = (eğim) * x1 şeklinde çizgi
slope_weights = -weights[0] / weights[1]
y_vals_weights = slope_weights * x_vals


# Scatter plot between two columns (change 'column_x' and 'column_y' to your actual column names)
plt.figure(figsize=(8, 6))
plt.scatter(df['Signal 1'], df['Signal 2'], color='purple', alpha=0.6, edgecolor='k')

# Grafiği çizin
plt.plot(x_vals, y_vals, label=f'Decision Boundary for F={F}', color="red")

plt.plot(x_vals, y_vals_weights, label=f'Decision Boundary for weights={weights}', color="blue")

method = 1
# Add labels and title
plt.xlabel('Signal 1')
plt.ylabel('Signal 2')
plt.title('Scatter Plot of Signal 1 vs Signal 2')
plt.savefig(f'line_plot_{method}.svg', format='svg', bbox_inches='tight')
# Display the plot
plt.show()
