import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 线性回归练习
def load_dataset(n):
    noise = np.random.rand(n)
    # print(noise)

    x = [ [i] for i in range(n)]
    # print(x)

    # y=0.5+1
    y = [ (0.5*x[i][0]+1.0+noise[i]) for i in range(n)]
    # print(y)

    return np.array(x),np.array(y)

x,y = load_dataset(10)
# print(x.shape,y.shape)
# plt.plot(x,y)
# plt.show()

# 画散点图
plt.scatter(x,y)
plt.title('data')
# plt.show()


linreg = LinearRegression()
linreg.fit(x,y)
y_pred = linreg.predict(x)

# plt.plot(x,y_pred)
# plt.show()

theta = np.hstack((linreg.intercept_,linreg.coef_))
print(theta)
print("Q=",linreg.coef_,", b=",linreg.intercept_)

x0 = np.ones((x.shape[0],1))
print(x0)

X = np.hstack((x0,x))
print(X)


# 正规方程求线性回归
def normal_equation(X,y):
    # theta = inv(X'X)X'y
    X_T_X = np.linalg.inv(X.T.dot(X))
    theta = np.dot(X_T_X,X.T).dot(y)
    return theta



theta = normal_equation(X,y)
print('normal ',theta)

y_pred = X.dot(theta)
print('normal',y_pred)
# plt.plot(x,y_pred)
# plt.show()


# 随机梯度下降方法求线性回归
def sgd(X,y,iters,lr):
    theta = np.zeros(X.shape[1])
    cnt = 0
    while True:
        for i in range(X.shape[0]):
            theta += lr * (y[i]-np.dot(X[i],theta)) * X[i]
            cnt += 1

        if cnt >= iters:
            break

    return theta

theta = sgd(X,y,1000,0.01)
print('normal ',theta)

y_pred = X.dot(theta)
print('normal',y_pred)
# plt.plot(x,y_pred)
# plt.show()

# 批梯度下降方法求线性回归
def bgd(X,y,iters,lr,bs):
    theta = np.zeros(X.shape[1])
    cnt = 0
    while True:
        for i in range(int(np.ceil(X.shape[0]/bs))):
            begin = i*bs
            end = X.shape[0] if i*bs + bs > X.shape[0] else i*bs +bs
            X_batch = X[begin: end]
            y_batch = y[begin: end]
            theta += lr * np.dot(X_batch.T,y_batch-np.dot(X_batch,theta))/X_batch.shape[0]
            cnt += 1

        if cnt >= iters:
            break

    return theta

theta = bgd(X,y,1000,0.01,3)
print('normal ',theta)

y_pred = X.dot(theta)
print('normal',y_pred)
plt.plot(x,y_pred)
plt.show()













