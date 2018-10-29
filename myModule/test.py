# encoding: utf-8

import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

def one_hot(y,class_num):
    one_hot_code = np.zeros((y.shape[0],class_num))
    for i in range(one_hot_code.shape[0]):
        one_hot_code[i][y[i]] = 1
    return one_hot_code


def softmax(x):
    sum_x = np.sum(np.exp(x),axis=1)

    softmax_out = x.copy()
    for i in range(x.shape[0]):
        softmax_out[i] = np.exp(x[i])/sum_x[i]
    return softmax_out


def cal_loss(x,y,theta):
    x0 = np.ones((x.shape[0],1))
    x = np.hstack((x0,x))
    p = softmax(x.dot(theta.T))
    y = one_hot(y,3)
    loss = -(y*np.log(p))
    loss = np.mean(loss)
    return loss


# 从load_iris中加载数据集
iris = load_iris()
x = iris.data
y = iris.target

# 把数据打乱
idx = list(range(x.shape[0]))
random.shuffle(idx)

# 划分训练数据与测试数据
train_num = int(x.shape[0]*0.8)
train_idx = idx[:train_num]
test_idx = idx[train_num:]

x_train = x[train_idx]
y_train = y[train_idx]
x_test = x[test_idx]
y_test = y[test_idx]

# 使用sklearn
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

theta = np.hstack((classifier.intercept_.reshape(-1,1),classifier.coef_))
print('theta:',theta)
loss = cal_loss(x_train,y_train,theta)
print('loss:',loss)

y_pred = classifier.predict(x_test)
print(y_test)
print(y_pred)




def predict(x,theta):
    x0 = np.ones((x.shape[0],1))
    x = np.hstack((x0,x))
    p = softmax(x.dot(theta.T))
    y_pred = np.argmax(p,axis=1)
    return y_pred

y_pred = predict(x_test,theta)
print(y_pred)