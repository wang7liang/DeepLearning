# encoding: utf-8

import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# 加载数据
def load_dataset():
    # 从load_iris中加载数据集
    iris = load_iris()
    x = iris.data
    y = iris.target

    # 逻辑回归是二分类，数据集中是3种类别，
    # 所以必须对数据集中的数据做处理，去掉其中一种类别的数据
    idx = []
    for i in range(y.shape[0]):
        if y[i] != 2:
            idx.append(i)

    x_b = x[idx]
    y_b = y[idx]

    # 把数据打乱
    idx = list(range(x_b.shape[0]))
    random.shuffle(idx)
    train_idx = idx[:80]
    test_idx = idx[80:]

    # 把处理过的数据集（100个），80个用于训练，20个用于测试
    x_train = x_b[train_idx]
    y_train = y_b[train_idx]
    x_test = x_b[test_idx]
    y_test = y_b[test_idx]

    return x_train,y_train,x_test,y_test

# sigmoid函数
def sigmoid(x):
    return(1/(1+np.exp(-x)))

# 计算loss
def cal_loss(x,y,theta):
    x0 = np.ones((x.shape[0], 1))
    x = np.hstack((x0, x))
    p = sigmoid(x.dot(theta))
    loss = -(y*np.log(p)+(1-y)*np.log(1-p))
    loss = np.mean(loss)
    return loss

# 用sklearn进行训练
def train_sklearn(x,y):
    classifier = LogisticRegression()
    classifier.fit(x, y)

    # coef_（系数）
    # print("w",classifier.coef_)
    # intercept_（截距）
    # print("b",classifier.intercept_)
    #
    theta = np.hstack((classifier.intercept_.reshape(-1,1),classifier.coef_))
    # print("theta",theta)

    return theta


# bgd
def train_bgd(x,y,bs,lr,iters):
    x0 = np.ones((x.shape[0],1))
    x = np.hstack((x0,x))
    theta = np.random.normal(0,0.1,x.shape[1])

    start = 0
    for i in range(iters):
        end = start+bs
        if end>x.shape[0]:
            end = x.shape[0]

        x_batch = x[start:end]
        y_batch = y[start:end]

        start = end
        if(start == x.shape[0]):
            start = 0

        y_pred = sigmoid(x_batch.dot(theta))
        grad = x_batch.T.dot((y_pred-y_batch))
        theta = theta - lr * grad

    return theta

# 预测
def predict(x,theta):
    x0 = np.ones((x.shape[0],1))
    x = np.hstack((x0,x))
    p = sigmoid(x.dot(theta))
    y_pred = (p>0.5).astype('int')
    return y_pred

x_train,y_train,x_test,y_test = load_dataset()

theta = train_bgd(x_train,y_train,32,0.01,1000)
print(theta)
loss = cal_loss(x_train,y_train,theta)
print(loss)
y_pred = predict(x_test,theta)
print(y_pred)
print(y_test)
