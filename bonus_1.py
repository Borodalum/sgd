import numpy as np
from sklearn.datasets import make_regression
from memory_profiler import memory_usage
import time


def loss(X, y, w, lambda_val, regularization=None):
    if regularization == 'L1':
        return np.sum((np.dot(X, w) - y) ** 2) / len(y) + lambda_val * np.sum(np.abs(w))
    elif regularization == 'L2':
        return np.sum((np.dot(X, w) - y) ** 2) / len(y) + lambda_val * np.sum(w ** 2)
    elif regularization == 'Elastic':
        return np.sum((np.dot(X, w) - y) ** 2) / len(y) + lambda_val * (0.5 * np.sum(w ** 2) + 0.5 * np.sum(np.abs(w)))
    else:
        return np.sum((np.dot(X, w) - y) ** 2) / len(y)


def gradient(X, y, w, lambda_val, regularization=None):
    if regularization == 'L1':
        return 2 * np.dot(X.T, (np.dot(X, w) - y)) / len(y) + lambda_val * np.sign(w)
    elif regularization == 'L2':
        return 2 * np.dot(X.T, (np.dot(X, w) - y)) / len(y) + lambda_val * w
    elif regularization == 'Elastic':
        return 2 * np.dot(X.T, (np.dot(X, w) - y)) / len(y) + lambda_val * (0.5 * w + 0.5 * np.sign(w))
    else:
        return 2 * np.dot(X.T, (np.dot(X, w) - y)) / len(y)


def SGD(X, y, h, lambda_val, batch_size=20, learning_rate_schedule=None, max_iter=1000, regularization=None):
    w = np.zeros(X.shape[1])  # инициализировать веса
    Q = loss(X, y, w, lambda_val, regularization)  # инициализировать оценку функционала

    for it in range(max_iter):
        if learning_rate_schedule is not None:
            h = learning_rate_schedule(it)

        batch_indices = np.random.choice(X.shape[0], size=batch_size, replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        eps = loss(X_batch, y_batch, w, lambda_val, regularization)  # вычислить потерю
        w = w - h * gradient(X_batch, y_batch, w, lambda_val,
                             regularization)  # обновить вектор весов в направлении антиградиента
        Q_new = lambda_val * eps + (1 - lambda_val) * Q  # оценить функционал

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
    lambda_val = 0.5

    regularizations = ['L1', 'L2', 'Elastic']

    for regularization in regularizations:
        start_mem = memory_usage(-1, interval=1, timeout=1)
        start_time = time.time()
        w = SGD(X, y, h, lambda_val, regularization=regularization)
        end_time = time.time()
        end_mem = memory_usage(-1, interval=1, timeout=1)
        print(f"Memory used by custom SGD with {regularization} regularization: ", max(end_mem) - max(start_mem), "MB")
        print(f"Time taken by custom SGD with {regularization} regularization: ", end_time - start_time, "seconds")
        print(f"Weights from custom SGD with {regularization} regularization:", w)
