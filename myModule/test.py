from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np


# 用sklearn进行训练并显示结果
def train_sklearn():
    x = np.eye(10)
    x[0][0] = 0
    print(x)

train_sklearn()

