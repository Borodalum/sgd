import numpy as np
from sklearn.datasets import make_regression
from memory_profiler import memory_usage
import time

def loss(X, y, w, λ, regularization=None):
    if regularization == 'L1':
        return np.sum((np.dot(X, w) - y) ** 2) / len(y) + λ * np.sum(np.abs(w))
    elif regularization == 'L2':
        return np.sum((np.dot(X, w) - y) ** 2) / len(y) + λ * np.sum(w ** 2)
    elif regularization == 'Elastic':
        return np.sum((np.dot(X, w) - y) ** 2) / len(y) + λ * (0.5 * np.sum(w ** 2) + 0.5 * np.sum(np.abs(w)))
    else:
        return np.sum((np.dot(X, w) - y) ** 2) / len(y)

def gradient(X, y, w, λ, regularization=None):
    if regularization == 'L1':
        return 2 * np.dot(X.T, (np.dot(X, w) - y)) / len(y) + λ * np.sign(w)
    elif regularization == 'L2':
        return 2 * np.dot(X.T, (np.dot(X, w) - y)) / len(y) + λ * w
    elif regularization == 'Elastic':
        return 2 * np.dot(X.T, (np.dot(X, w) - y)) / len(y) + λ * (0.5 * w + 0.5 * np.sign(w))
    else:
        return 2 * np.dot(X.T, (np.dot(X, w) - y)) / len(y)

def SGD(X, y, h, λ, batch_size=20, learning_rate_schedule=None, max_iter=1000, regularization=None):
    w = np.zeros(X.shape[1])  # инициализировать веса
    Q = loss(X, y, w, λ, regularization)  # инициализировать оценку функционала

    for iter in range(max_iter):
        if learning_rate_schedule is not None:
            h = learning_rate_schedule(iter)

        batch_indices = np.random.choice(X.shape[0], size=batch_size, replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        ε = loss(X_batch, y_batch, w, λ, regularization)  # вычислить потерю
        w = w - h * gradient(X_batch, y_batch, w, λ, regularization)  # обновить вектор весов в направлении антиградиента
        Q_new = λ * ε + (1 - λ) * Q  # оценить функционал

        if np.abs(Q_new - Q) < 1e-6:  # проверить сходимость
            break

        Q = Q_new

    return w

def generate_data(n_samples=100, n_features=5, noise=0.1):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
    return X, y

if __name__ == '__main__':
    X, y = generate_data()
    h = 0.01
    λ = 0.5

    regularizations = ['L1', 'L2', 'Elastic']

    for regularization in regularizations:
        start_mem = memory_usage(-1, interval=1, timeout=1)
        start_time = time.time()
        w = SGD(X, y, h, λ, regularization=regularization)
        end_time = time.time()
        end_mem = memory_usage(-1, interval=1, timeout=1)
        print(f"Memory used by custom SGD with {regularization} regularization: ", max(end_mem) - max(start_mem), "MB")
        print(f"Time taken by custom SGD with {regularization} regularization: ", end_time - start_time, "seconds")
        print(f"Weights from custom SGD with {regularization} regularization:", w)