import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_contour_sns(X, y):
    # Создаем DataFrame из данных
    data = pd.DataFrame(data=np.c_[X[:, :2], y], columns=['Feature 1', 'Feature 2', 'Target'])
    # Рисуем график с линиями уровня
    sns.kdeplot(data=data, x='Feature 1', y='Feature 2', fill=True)
    plt.show()

    # Объёмный график
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')
    plt.show()
