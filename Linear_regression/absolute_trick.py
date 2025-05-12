def absolute_trick(base_price,price_per_room,num_room,price,learning_rate):
    predicted_price = base_price + price_per_room*num_room
    if price > predicted_price:
        price_per_room += learning_rate * num_room
        base_price += learning_rate
    else:
        price_per_room -= learning_rate * num_room
        base_price -= learning_rate

    return base_price,price_per_room


