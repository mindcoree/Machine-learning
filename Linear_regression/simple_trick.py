import numpy as np
import matplotlib.pyplot as plt

features = np.array([1,2,3,5,6,7]) # признаки или входные данные (X)
labels = np.array([155,194,244,356,407,448]) # метки или целевые значения (Y)
# features количество комнат
# labels цена за дом

fig = plt.figure(figsize=(5,5))
ax = fig.subplots()
ax.set(title="Диаграмма рассевания",xlabel="number of room",ylabel="prices")
ax.scatter(features,labels,color='r')

plt.show()

# creation of a straight line for forecast
def simple_trick(base_price,price_per_room,num_room,price):
    small_random_1 = np.random.rand()*0.1
    small_random_2 = np.random.rand()*0.1
    predicted_price = base_price + num_room*price_per_room # y = ax + b
    #1 если точка находится справа и над прямой
    if price > predicted_price and num_room > 0:
        base_price += small_random_1 # Y пересечение
        price_per_room += small_random_2 # угол наклона
    #2 если точка находиться слева и над прямой
    elif price > predicted_price and num_room < 0:
        base_price += small_random_1
        price_per_room -= small_random_2
    #3 если точка находится справа и под прямой
    elif price < predicted_price and num_room > 0:
        base_price -= small_random_1
        price_per_room -= small_random_2
    #4 если точка находиться слева и под прямой
    elif price < predicted_price and num_room < 0:
        base_price -= small_random_1
        price_per_room += small_random_2

    return base_price,price_per_room

