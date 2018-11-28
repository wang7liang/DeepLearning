# encoding: utf-8

import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# 准备数据
def getData():
    target = load_iris().target
    data = load_iris().data

    idx = []
    for i in range(target.shape[0]):
        if target[i]!=2:
            idx.append(i)

    target_b = target[idx]
    data_b = data[idx]

    idx = range(100)
    random.shuffle(idx)

    target_b = target_b[idx]
    data_b = data_b[idx]

    trainCount = int(target_b.shape[0]*0.8)
    y_train = target_b[:trainCount]
    x_train = data_b[:trainCount]
    y_test = target_b[trainCount:]
    x_test = data_b[trainCount:]

    return x_train,y_train,x_test,y_test

# sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

# sigmoid导函数
def sigmoid_d(z):
    return sigmoid(z)*(1-sigmoid(z))

def train(x,y):
    # 初始化
    a = 0.01
    m = y.shape[0]
    W1 = np.random.randn(2,4)*0.01
    b1 = np.zeros((2,1))
    W2 = np.random.randn(1,2)*0.01
    b2 = np.zeros((1,1))

    for i in range(10000):
        # 正向传播
        Z1 = W1.dot(x)+b1
        A1 = sigmoid(Z1)
        Z2 = W2.dot(A1)+b2
        A2 = sigmoid(Z2)

        # 反向传播
        dZ2 = A2-y
        dW2 = dZ2.dot(A1.T)/m
        db2 = np.sum(dZ2,axis=1,keepdims=True)
        dZ1 = np.dot(W2.T,dZ2)*sigmoid_d(Z1)
        dW1 = dZ1.dot(x.T)/m
        db1 = np.sum(dZ1,axis=1,keepdims=True)

        # 梯度下降
        W1-=a*dW1
        b1-=a*db1
        W2-=a*dW2
        b2-=a*db2

    return W1,b1,W2,b2

def pred(x_test,W1,b1,W2,b2):
    Z1 = W1.dot(x_test) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    y_pred = (A2 > 0.5).astype('int')
    return y_pred

# 开始
x_train,y_train,x_test,y_test = getData()
W1,b1,W2,b2 = train(x_train.T,y_train)
y_pred = pred(x_test.T,W1,b1,W2,b2)

print(y_pred)
print(y_test.reshape(1,-1))
