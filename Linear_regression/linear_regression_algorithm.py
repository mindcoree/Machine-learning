import numpy as np
import matplotlib.pyplot as plt

# Фиксируем генератор случайных чисел для воспроизводимости
np.random.seed(0)

# Алгоритм корректировки коэффициентов модели по MSE (square trick)
def square_trick(base_price, price_per_room, num_room, price, learning_rate):
    predicted_price = base_price + price_per_room * num_room  # предсказанная цена: y = a * x + b
    # корректировка параметров по градиенту
    base_price += learning_rate * (price - predicted_price)  # обновляем b
    price_per_room += learning_rate * num_room * (price - predicted_price)  # обновляем a
    return base_price, price_per_room

# Функция отрисовки прямой линии: y = slop * x + y_intercept
def draw_line(slop, y_intercept, color='gray',linewidth=0.7):
    x = np.linspace(0, 8, 100)
    y = y_intercept + slop * x
    plt.plot(x, y, color=color, linestyle='--',linewidth=linewidth)

# Отрисовка исходных точек
def plot_points(x, y):
    plt.scatter(x, y)
    plt.xlabel('Number of rooms')
    plt.ylabel('Prices')

# Основная функция обучения модели линейной регрессии
def linear_regression(features, labels, learning_rate=0.01, epochs=1000):
    # Случайная инициализация коэффициентов
    price_per_room = np.random.rand()
    base_price = np.random.rand()

    for epoch in range(epochs):
        draw_line(price_per_room, base_price)  # рисуем текущую прямую на каждом шаге
        idx = np.random.randint(0, len(features))  # случайный индекс точки из данных
        num_room = features[idx]
        price = labels[idx]
        # обновляем параметры модели
        base_price, price_per_room = square_trick(
            base_price,
            price_per_room,
            num_room,
            price,
            learning_rate=learning_rate
        )

    # финальная модель — жирная чёрная линия
    draw_line(price_per_room, base_price, color='black')
    # рисуем точки
    plot_points(features, labels)
    # выводим итоговые параметры
    print(f"Price per room: {price_per_room:.2f}")
    print(f"Base price: {base_price:.2f}")
    return base_price, price_per_room

# Наши входные данные (features — количество комнат, labels — цены)
features = np.array([1, 2, 3, 5, 6, 7])
labels = np.array([155, 194, 244, 356, 407, 448])

# Запуск обучения модели
linear_regression(features, labels, learning_rate=0.01, epochs=100000)

# Показываем финальный график
plt.title("Linear Regression via Square Trick")
plt.grid()
plt.show()
