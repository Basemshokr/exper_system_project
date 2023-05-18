import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'D:\\project\\exam.txt'
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])
print(' data = ')
print(data.head(10))
print('===============')
print('data.describe =')
print(data.describe())
positive = data[data["admitted"].isin([1])]
negative = data[data["admitted"].isin([0])]
print('admitted student \n', positive)
print('==============')
print('non admitted student', negative)
print('============')
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='admitted')
ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='non admitted')
ax.legend()
ax.set_xlabel('exam1 Score')
ax.set_ylabel('exam2 Score')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(nums, sigmoid(nums), 'r')


def cost(thetav, xv, yv):
    thetav = np.matrix(thetav)
    xv = np.matrix(xv)
    yv = np.matrix(yv)
    first = np.multiply((-yv), np.log(sigmoid(xv * thetav.T)))
    second = np.multiply((1 - yv), np.log(1 - sigmoid(xv * thetav.T)))
    return np.sum(first - second) / (len(xv))


data.insert(0, 'ones', 1)
print('new data \n', data)
print('=======')
cols = data.shape[1]
x = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]
print('x\n', x)
print('=======================')
print('y\n', y)
print('=======================')
x = np.array(x.values)
y = np.array(y.values)
theta = np.zeros(3)
print('x \n', x)
print('========')
print('y \n', y)
print('=========')
print('x.shape =', x.shape)
print('theta.shape', theta.shape)
print('x.shape', x.shape)
thiscost = cost(theta, x, y)
print('========')
print('cost=', thiscost)


def gradient(thetav, xv, yv):
    thetav = np.matrix(thetav)
    xv = np.matrix(xv)
    yv = np.matrix(yv)
    parameters = int(thetav.ravel().shape[1])

    grad = np.zeros(parameters)
    error = sigmoid(xv * thetav.T) - yv

    for i in range(parameters):
        term = np.multiply(error, xv[:, i])
        grad[i] = np.sum(term) / len(xv)

    return grad


import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
print('=======')
print('result=', result)
costafteroptimize = cost(result[0], x, y)
print()
print('cost after optimize =', costafteroptimize)
print()


def predict(theta, x):
    probability = sigmoid(x * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


theta_min = np.matrix(result[0])
prediction = predict(theta_min, x)
print('new predict', prediction)
print()
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(prediction, y)]
accuracy = (sum(map(int, correct)) / len(correct))
print('accuracy={0}%'.format(accuracy))