import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
method = 3
data = pd.read_csv(f'../../Performance Function/weight_trajectories_performance_{method}.csv')

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(data['mean_w1'], label=r'$w_1(n)$', color='black', linewidth=1)
plt.plot(data['mean_w2'], label=r'$w_2(n)$', color='black', linestyle='--', linewidth=1)

# Adding labels and customizing plot
plt.title("Weight Trajectories of the Algorithm")
plt.xlabel("Number of Samples")
plt.ylabel("Weight Trajectories")
plt.legend()
plt.ylim(-1, 1)  # Adjusting y-limits based on the target plot's range
plt.xlim(0, 1000)  # Adjusting x-limits to match 1000 iterations

# Adding final weight annotations
plt.text(1000, data['mean_w1'].iloc[-1], f"{data['mean_w1'].iloc[-1]:.3f}", verticalalignment='center')
plt.text(1000, data['mean_w2'].iloc[-1], f"{data['mean_w2'].iloc[-1]:.3f}", verticalalignment='center')

# Show grid and plot
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(f'weight_trajectory_plot_algorithm_{method}.svg', format='svg', bbox_inches='tight')  # PDF olarak kaydeder

plt.show()
