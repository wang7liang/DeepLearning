import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 生成训练数据方法
def load_dataset(n):
    noise = np.random.rand(n)
    x = [ [i] for i in range(n) ]
    y = [ (0.5*x[i][0]+1.0+noise[i]) for i in range(n)]
    return np.array(x),np.array(y)


# 用sklearn进行训练并显示结果
def train_sklearn():
    # 生成训练数据
    x,y = load_dataset(10)

    # 训练
    linreg = LinearRegression()
    linreg.fit(x, y)

    # 取得结果
    y_pred = linreg.predict(x)

    # 画散点图
    plt.scatter(x,y)

    # 画出结果直线
    plt.plot(x,y_pred)
    plt.show()

    # Theta = {b,w1,w2,...wn}
    # intercept_:偏移量b
    # coef_:w
    #theta = np.hstack((linreg.intercept_,linreg.coef_))

    # X = {1,x1,x2,...xn}
    #X = np.hstack((np.ones((x.shape[0],1)),x))

    # y_pred = X.dot(theta)


# 线性回归——正规方程形式
def train_normal():
    # 生成训练数据
    x, y = load_dataset(10)

    X = np.hstack((np.ones((x.shape[0], 1)), x))

    # theta = inv(X`X)X`y
    X_T_X = np.linalg.inv(X.T.dot(X))
    theta = np.dot(X_T_X,X.T.dot(y))

    y_pred = X.dot(theta)

    # 画散点图
    plt.scatter(x, y)

    # 画出结果直线
    plt.plot(x, y_pred)
    plt.show()


# 线性回归——随机梯度下降
def train_sgd():
    # 生成训练数据
    x, y = load_dataset(10)

    X = np.hstack((np.ones((x.shape[0], 1)), x))

    lr = 0.01;
    theta = np.zeros(X.shape[1])

    cnt = 0
    while True:
        for i in range(X.shape[0]):
            theta += lr*(y[i]-np.dot(X[i],theta)) *X[i]
            cnt+=1
        if cnt >=100:
            break

    y_pred = X.dot(theta)

    # 画散点图
    plt.scatter(x, y)

    # 画出结果直线
    plt.plot(x, y_pred)
    plt.show()



# 线性回归——批机梯度下降
def train_bgd():
    # 生成训练数据
    x, y = load_dataset(10)

    X = np.hstack((np.ones((x.shape[0], 1)), x))

    bs = 3
    lr = 0.01;
    theta = np.zeros(X.shape[1])

    cnt = 0

    while True:
        for i in range(int(np.ceil(X.shape[0]/bs))):
            begin = i * bs
            end = X.shape[0] if i * bs + bs > X.shape[0] else i * bs + bs

            X_batch = X[begin: end]
            y_batch = y[begin: end]

            theta += lr * np.dot(X_batch.T, y_batch - np.dot(X_batch, theta)) / X_batch.shape[0]

            cnt+=1
        if cnt >=1000:
            break

    y_pred = X.dot(theta)

    # 画散点图
    plt.scatter(x, y)

    # 画出结果直线
    plt.plot(x, y_pred)
    plt.show()

train_bgd()