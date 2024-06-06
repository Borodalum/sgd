import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_contour_sns(X, y, w):
    # Создаем DataFrame из данных
    data = pd.DataFrame(data=np.c_[X[:, :2], y], columns=['Feature 1', 'Feature 2', 'Target'])
    # Рисуем график с линиями уровня
    sns.kdeplot(data=data, x='Feature 1', y='Feature 2', fill=True)
    plt.show()

    # Плоский график с аппроксимационной прямой
    sns.regplot(x='Feature 1', y='Feature 2', data=data)
    plt.show()

    # Объёмный график c аппроксимационной плоскостью
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['Feature 1'], data['Feature 2'], data['Target'], c='b', marker='o')

    # Создаем сетку значений для X и Y
    x_surf = np.arange(X[:, 0].min(), X[:, 0].max(), 0.01)
    y_surf = np.arange(X[:, 1].min(), X[:, 1].max(), 0.01)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)

    # Вычисляем соответствующие значения Z
    z_surf = w[0] * x_surf + w[1] * y_surf
    ax.plot_surface(x_surf, y_surf, z_surf, color='deepskyblue', alpha=0.5)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')
    plt.show()
