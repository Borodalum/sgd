import tensorflow as tf
from sklearn.datasets import make_regression
import numpy as np
from memory_profiler import memory_usage
import time
import pandas as pd

import plotting


def loss(X, y, w):
    return (np.sum((np.dot(X, w) - y) ** 2) / len(y))


def gradient(X, y, w):
    return 2 * np.dot(X.T, (np.dot(X, w) - y)) / len(y)


def SGD(X, y, h, λ, batch_size=20, learning_rate_schedule=None, max_iter=1000):
    w = np.zeros(X.shape[1])  # инициализировать веса
    Q = loss(X, y, w)  # инициализировать оценку функционала

    for iter in range(max_iter):
        if learning_rate_schedule is not None:
            h = learning_rate_schedule(iter)

        batch_indices = np.random.choice(X.shape[0], size=batch_size, replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        ε = loss(X_batch, y_batch, w)  # вычислить потерю
        w = w - h * gradient(X_batch, y_batch, w)  # обновить вектор весов в направлении антиградиента
        Q_new = λ * ε + (1 - λ) * Q  # оценить функционал

        if np.abs(Q_new - Q) < 1e-6:  # проверить сходимость
            break

        Q = Q_new

    return w


def step_decay_schedule(initial_lr=0.1, decay_factor=0.5, step_size=10):

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return schedule


def generate_data(n_samples=100, n_features=5, noise=0.1):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
    return X, y


if __name__ == '__main__':
    X, y = generate_data()
    h = 0.01
    λ = 0.5

    df = pd.DataFrame(columns=['Method', 'Memory Used', 'Time Taken', 'Weights'])

    # Использование вашей функции SGD
    start_mem = memory_usage(-1, interval=1, timeout=1)
    start_time = time.time()
    w = SGD(X, y, h, λ)
    end_time = time.time()
    end_mem = memory_usage(-1, interval=1, timeout=1)
    print("Memory used by custom SGD: ", max(end_mem) - max(start_mem), "MB")
    print("Time taken by custom SGD: ", end_time - start_time, "seconds")
    print("Weights from custom SGD:", w)
    df.loc[0] = ['Custom SGD', max(end_mem) - max(start_mem), end_time - start_time, w]

    # Использование нашей функции с другим batch_size
    start_mem = memory_usage(-1, interval=1, timeout=1)
    start_time = time.time()
    w = SGD(X, y, h, λ, batch_size=2)
    end_time = time.time()
    end_mem = memory_usage(-1, interval=1, timeout=1)
    print("Memory used by custom SGD with batch_size=2: ", max(end_mem) - max(start_mem), "MB")
    print("Time taken by custom SGD with batch_size=2: ", end_time - start_time, "seconds")
    print("Weights from custom SGD with batch_size=2:", w)
    df.loc[1] = ['Custom SGD with batch_size=2', max(end_mem) - max(start_mem), end_time - start_time, w]

    # Использование нашей функции с learning rate schedule
    step_schedule = step_decay_schedule(initial_lr=0.1, decay_factor=0.5, step_size=10)
    start_mem = memory_usage(-1, interval=1, timeout=1)
    start_time = time.time()
    w = SGD(X, y, h, λ, learning_rate_schedule=step_schedule)
    end_time = time.time()
    end_mem = memory_usage(-1, interval=1, timeout=1)
    print("Memory used by custom SGD with step decay learning rate schedule: ", max(end_mem) - max(start_mem), "MB")
    print("Time taken by custom SGD with step decay learning rate schedule: ", end_time - start_time, "seconds")
    print("Weights from custom SGD with step decay learning rate schedule:", w)
    df.loc[2] = ['Custom SGD with step decay learning rate schedule', max(end_mem) - max(start_mem), end_time - start_time, w]

    # Использование SGD из TensorFlow
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)
    ])

    optimizers = [
        tf.keras.optimizers.SGD(learning_rate=h),
        tf.keras.optimizers.SGD(learning_rate=h, momentum=0.9),
        tf.keras.optimizers.SGD(learning_rate=h, momentum=0.9, nesterov=True),
        tf.keras.optimizers.Adagrad(learning_rate=h),
        tf.keras.optimizers.RMSprop(learning_rate=h),
        tf.keras.optimizers.Adam(learning_rate=h)
    ]

    for optimizer in optimizers:
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        start_mem = memory_usage(-1, interval=1, timeout=1)
        start_time = time.time()
        model.fit(X, y, epochs=1000, verbose=0)
        end_time = time.time()
        end_mem = memory_usage(-1, interval=1, timeout=1)
        print(f"Memory used by {optimizer.get_config()['name']}: ", max(end_mem) - max(start_mem), "MB")
        print(f"Time taken by {optimizer.get_config()['name']}: ", end_time - start_time, "seconds")
        print(f"Weights from {optimizer.get_config()['name']}:", model.get_weights()[0].flatten())
        df.loc[len(df)] = [optimizer.get_config()['name'], max(end_mem) - max(start_mem), end_time - start_time, model.get_weights()[0].flatten()]

    # Рисуем график с линиями уровня
    plotting.plot_contour_sns(X, y, w)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df)