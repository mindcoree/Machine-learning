import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import turicreate as tc
random.seed(0)

# Our original polynomial is -x^2+x+15
coefs = [15,1,-1]

def polynomial(coefs, x):
    n = len(coefs)
    return sum([coefs[i]*x**i for i in range(n)])

def draw_polynomial(coefs):
    n = len(coefs)
    x = np.linspace(-5, 5, 1000)
    plt.ylim(-20,20)
    plt.plot(x, sum([coefs[i]*x**i for i in range(n)]), linestyle='-', color='black')

draw_polynomial(coefs)

X = []
Y = []
for i in range(40):
    x = random.uniform(-5,5)
    #random.gauss шум из нормального распределения с математическим ожиданием 0 и стандартным отклонением 2.
    y = polynomial(coefs, x) + random.gauss(0,2)
    X.append(x)
    Y.append(y)

plt.scatter(X, Y)
#draw_polynomial(coefs)
plt.show()
data = tc.SFrame({'x':X, 'y':Y})
# Adding columns to our dataset corresponding to $x^2, x^3,..., x^{200}$
for i in range(2,200):
    string = 'x^'+str(i)
    data[string] = data['x'].apply(lambda x:x**i)

train, test = data.random_split(.8, seed=0)

def display_results(model):
    coefs = model.coefficients
    print("Training error (rmse):", model.evaluate(train)['rmse'])
    print("Testing error (rmse):", model.evaluate(test)['rmse'])
    plt.scatter(train['x'], train['y'], marker='o')
    plt.scatter(test['x'], test['y'], marker='^')
    draw_polynomial(coefs['value'])
    plt.show()
    print("Polynomial coefficients")
    print(coefs['name', 'value'])


model_no_reg = tc.linear_regression.create(
    train, target='y', l1_penalty=0.0, l2_penalty=0.0, verbose=False, validation_set=None)


model_L2_reg = tc.linear_regression.create(
    train, target='y', l1_penalty=0.0, l2_penalty=0.1, verbose=False, validation_set=None)

model_L1_reg = tc.linear_regression.create(
    train, target='y', l1_penalty=0.1, l2_penalty=0.0, verbose=False, validation_set=None)

display_results(model_no_reg)
display_results(model_L1_reg)
display_results(model_L2_reg)

predictions = test['x', 'y']
predictions['No reg'] = model_no_reg.predict(test)
predictions['L1 reg'] = model_L1_reg.predict(test)
predictions['L2 reg'] = model_L2_reg.predict(test)
