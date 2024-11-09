import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_d_q_n(training_data, F):
    d_q_n_values = []
    for x_n in training_data:
        d_n = np.dot(F, x_n)
        d_q_n = 1 if d_n > 0 else -1
        d_q_n_values.append(d_q_n)
    return np.array(d_q_n_values)  # Listeyi numpy array olarak döndür

def calculate_y_q_n(training_data, W):
    y_q_n_values = []
    for x_n in training_data:
        y_n = np.dot(W, x_n)
        y_q_n = 1 if y_n > 0 else -1
        y_q_n_values.append(y_q_n)
    return np.array(y_q_n_values)  # Listeyi numpy array olarak döndür

if __name__ == '__main__':
    F = [-1, -1]
    df = pd.read_csv('../../Data Generation/jointly_gaussian_data.csv')
    X = df.values  # X veri noktaları


    # d_q_values'in boyutunu X veri noktalarına göre ayarla
    d_q_values = calculate_d_q_n(X, F)

    # Boyut uyumsuzluğunu kontrol et
    assert len(d_q_values) == X.shape[0], "d_q_values, X'in satır sayısıyla eşleşmiyor!"

    weight1_values = np.linspace(-1, 1, 50)  # weight1 için aralık
    weight2_values = np.linspace(-1, 1, 50)  # weight2 için aralık

    perf = 3

    performance_values = np.zeros((len(weight1_values), len(weight2_values)))
    # Her bir weight1, weight2 çifti için performans değerini hesapla
    for i, w1 in enumerate(weight1_values):
        for j, w2 in enumerate(weight2_values):
            weights = np.array([w1, w2])
            y_q_values = calculate_y_q_n(X, weights)

            # y(n) = X * weights hesapla
            y_values = X @ weights
            # Performans fonksiyonu hesapla: (2|y(n)| - 2 * d_q(n) * y(n))
            if perf == 1:
                performance = np.mean(2 * np.abs(y_values) - 2 * d_q_values * y_values)
            elif perf == 2:
                performance = np.mean((d_q_values - y_q_values) ** 2)
            elif perf == 3:
                performance = np.mean((y_values - d_q_values) ** 2)
            # Hesaplanan performansı matrise yerleştir
            performance_values[i, j] = min(performance, 2)

    # Surface plot oluşturma ve kaydetme
    fig = plt.figure(figsize=(12, 9), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    W1, W2 = np.meshgrid(weight1_values, weight2_values)
    ax.plot_surface(W1, W2, performance_values.T, cmap='plasma', edgecolor='k')
    # ax.plot_wireframe(W1, W2, performance_values.T, color='black')
    ax.view_init(elev=20, azim=235)
    ax.dist = 7
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('Performance')
    ax.set_title('Performance Surface Plot')

    # Grafiği kaydetme
    plt.savefig(f'performance_{perf}_surface_plot.svg', format='svg', bbox_inches='tight')  # PDF olarak kaydeder

    plt.show()
