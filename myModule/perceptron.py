import numpy as np
from matplotlib import pyplot as mpt
import input_data



# 单层感知机简单形式
# ps: 所有绘图方法只有当样本集是二维，标签集是一维时才生效
class MyPerceptron:

    # 样本集
    sampleSet = 0
    # 标签集
    labelSet = 0
    # 错误次数集
    wrongSet = list()
    # gram矩阵
    gramMat = 0


    # 训练目标参数
    w = [0,0];
    # 训练目标参数
    b = 0;
    # 学习率
    ln = 1;

    # 构造函数，传入样本集与标签集
    def __init__(self,sampleSet, labelSet):
        self.sampleSet = sampleSet
        self.labelSet = labelSet
        pass


    # 训练
    def train(self,n=1):
        for j in range(n):

            # 一轮训练
            err_count = 0
            for i in range(len(self.labelSet)):
                if self.labelSet[i] != np.sign(np.dot(self.w,self.sampleSet[i])+self.b):
                    self.w+=self.ln*self.labelSet[i]*self.sampleSet[i]
                    self.b+=self.labelSet[i]
                    err_count+=1
            if err_count == 0 :
                break

    # 取得gram矩阵
    def gram(self):
        self.gramMat = np.dot(self.sampleSet , self.sampleSet.T)

    # 用对偶形式训练
    def train_fast(self,n=1):

        for i in range(len(self.labelSet)):
            self.wrongSet.append(0)

        for j in range(n):

            # 一轮训练
            err_count = 0
            for i in range(len(self.labelSet)):

                # 累计w,b
                c_tmp = 0
                b_tmp = 0
                for k in range(len(self.labelSet)):
                    c_tmp+=self.wrongSet[k]*self.ln*self.labelSet[k]*self.gramMat[i][k]
                    b_tmp+=self.wrongSet[k]*self.ln*self.labelSet[k]

                if self.labelSet[i] != np.sign(c_tmp+b_tmp):
                    self.wrongSet[i]+=1
                    err_count+=1
            if err_count == 0 :
                break

        # 累计w,b
        for k in range(len(self.labelSet)):
            self.w += self.wrongSet[k] * self.ln * self.labelSet[k] * self.sampleSet[k]
            self.b += self.wrongSet[k] * self.ln * self.labelSet[k]


    # 测试
    def test(self,testSet):
        resultSet = list()
        for i in range(len(testSet)):
            resultSet.append(np.sign(np.dot(self.w,testSet[i])+self.b))
        print(resultSet)

    # 显示初始状态
    def show_init(self):
        if self.is_can_show():
            for i in range(len(self.labelSet)):
                if self.labelSet[i] >=0:
                    mpt.plot(self.sampleSet[i][0],self.sampleSet[i][1],'bo')
                else:
                    mpt.plot(self.sampleSet[i][0], self.sampleSet[i][1], 'ro')
            mpt.show()

    # 显示训练结果
    def show_result(self):
        print(self.w,":",self.b)

        if self.is_can_show():
            for i in range(len(self.labelSet)):
                if self.labelSet[i] >=0:
                    mpt.plot(self.sampleSet[i][0],self.sampleSet[i][1],'bo')
                else:
                    mpt.plot(self.sampleSet[i][0], self.sampleSet[i][1], 'ro')

            x = [0,10]
            y = [0,0]
            for i in range(len(x)):
                y[i] = (-self.w[0]*x[i]-self.b)/(self.w[1]+1e-6)
            mpt.plot(x,y)
            mpt.show()

    # 显示训练结果(对偶)
    def show_result_fast(self):
        print(self.wrongSet)

        if self.is_can_show():
            for i in range(len(self.labelSet)):
                if self.labelSet[i] >= 0:
                    mpt.plot(self.sampleSet[i][0], self.sampleSet[i][1], 'bo')
                else:
                    mpt.plot(self.sampleSet[i][0], self.sampleSet[i][1], 'ro')

            x = [0, 10]
            y = [0, 0]

            # 累计w,b
            c_tmp = 0
            b_tmp = 0
            for k in range(len(self.labelSet)):
                c_tmp += self.wrongSet[k] * self.ln * self.labelSet[k] * self.gramMat[i][k]
                b_tmp += self.wrongSet[k] * self.ln * self.labelSet[k]



            for i in range(len(x)):
                y[i] = (-self.w[0] * x[i] - self.b) / (self.w[1] + 1e-6)
            mpt.plot(x, y)
            mpt.show()

    # 检查绘图程序是否可执行
    # 所有绘图方法只有当样本集是二维，标签集是一维时才生效
    def is_can_show(self):
        # print(np.shape(self.sampleSet),":",type(np.shape(self.sampleSet)))
        # print(np.shape(self.labelSet),":",type(np.shape(self.labelSet)))
        if len(np.shape(self.sampleSet)) !=2 :
            return False
        if len(np.shape(self.labelSet)) !=1 :
            return False
        return True




# 开始
# 定义训练集
x = np.array([[2,2],[4,5],[7,1],[8,2],[9,2.1]])
y = np.array([1,1,-1,1,-1])

# 生成感知机
myPerceptron = MyPerceptron(x,y)
# myPerceptron.show_init()
# myPerceptron.train(100)

myPerceptron.gram()
myPerceptron.train_fast(100)
myPerceptron.show_result()




# testSet = np.array([[2,2],[4,5],[7,1],[8,2]])
# myPerceptron.test(testSet)

