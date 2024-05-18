import tensorflow as tf
from sklearn.datasets import make_regression
import numpy as np


def loss(X, y, w):
    return np.sum((np.dot(X, w) - y) ** 2) / len(y)


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

    # Использование вашей функции SGD
    w = SGD(X, y, h, λ)
    print("Weights from custom SGD:", w)

    # Использование нашей функции с другим batch_size
    w = SGD(X, y, h, λ, batch_size=2)
    print("Weights from custom SGD with batch_size=2:", w)

    # Использование нашей функции с learning rate schedule
    step_schedule = step_decay_schedule(initial_lr=0.1, decay_factor=0.5, step_size=10)
    w = SGD(X, y, h, λ, learning_rate_schedule=step_schedule)
    print("Weights from custom SGD with step decay learning rate schedule:", w)

    # Использование SGD из TensorFlow
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=h), loss='mean_squared_error')
    model.fit(X, y, epochs=1000, verbose=0)

    print("Weights from TensorFlow SGD:", model.get_weights()[0].flatten())

