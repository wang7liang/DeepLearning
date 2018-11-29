import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.data import Dataset
from sklearn import metrics
import math



def draw_scatter():
    print(min(np.array(validation_samples['latitude'])),max(np.array(validation_samples['latitude'])))
    print(min(np.array(validation_samples['longitude'])),max(np.array(validation_samples['longitude'])))

    print(min(np.array(training_samples['latitude'])),max(np.array(training_samples['latitude'])))
    print(min(np.array(training_samples['longitude'])),max(np.array(training_samples['longitude'])))


    plt.figure(
        figsize=(13,8)
    )
    ax = plt.subplot(1,2,1)
    ax.set_title('Validation data')
    ax.set_autoscaley_on(False)
    ax.set_ylim([34,42])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-124,-118])
    plt.scatter(validation_samples['longitude'],
                validation_samples['latitude'],
                cmap='coolwarm',
                c=validation_targets/validation_targets.max())


    ax = plt.subplot(1,2,2)
    ax.set_title('Train data')
    ax.set_autoscaley_on(False)
    ax.set_ylim([32,42])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-124,-114])
    plt.scatter(training_samples['longitude'],
                training_samples['latitude'],
                cmap='coolwarm',
                c=training_targets / training_targets.max()
                )


def input_fn(training_samples, training_targets, batch_size=1, shuffle=True, num_epochs=None):
    ds = Dataset.from_tensor_slices((
        {key: np.array(value).reshape(-1, 1) for key, value in dict(training_samples).items()},
        np.array(training_targets).reshape(-1, 1)
    ))
    ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds.shuffle(training_samples.shape[0])

    inputs, label = ds.make_one_shot_iterator().get_next()

    return inputs,label


def train_model(training_samples, training_targets, validation_samples, validation_targets, learning_rate, batch_size, steps):
    # 定义输入特征列
    feature_columns = [tf.feature_column.numeric_column(item) for item in training_samples]

    # 定义梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # 定义线性回归模型
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=optimizer
    )

    training_rmses = []
    validation_rmses = []
    periods = 10
    steps_per_periods = steps/periods
    for period in range(0,periods):
        # 训练
        linear_regressor.train(
            input_fn=lambda: input_fn(training_samples, training_targets, batch_size),
            steps=steps_per_periods
        )

        # 训练集预测
        training_predictions = linear_regressor.predict(
            input_fn=lambda: input_fn(training_samples, training_targets, shuffle=False, num_epochs=1)
        )
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        # 验证集预测
        validation_predictions = linear_regressor.predict(
            input_fn=lambda: input_fn(validation_samples, validation_targets, shuffle=False, num_epochs=1)
        )
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # 训练集均方误差
        training_mse = metrics.mean_squared_error(training_predictions,training_targets)
        training_rmse = math.sqrt(training_mse)
        training_rmses.append(training_rmse)

        # 验证集均方误差
        validation_mse = metrics.mean_squared_error(validation_predictions,validation_targets)
        validation_rmse = math.sqrt(validation_mse)
        validation_rmses.append(validation_rmse)

        # 打印
        print('training_rmse: %0.3f, validation_rmse: %0.3f' % (training_rmse, validation_rmse))


    # 画图
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title('RMSE vs. Periods')
    plt.tight_layout()
    plt.plot(training_rmses,label='training')
    plt.plot(validation_rmses,label='validation')

    return linear_regressor;


# 对输入特征进行预处理
def preprocess_features(dataframe):
    # 暂时只使用经纬度对房价进行估计
    selected_features = dataframe[[
        # 'longitude',
        'latitude',
        # 'housingMedianAge',
        # 'totalRooms',
        # 'totalBedrooms',
        # 'population',
        # 'households',
        'medianIncome'
    ]]
    selected_features = selected_features.copy()
    return selected_features


# 对标签进行预处理
def preprocess_targets(dataframe):
    output_targets = dataframe[['medianHouseValue']];
    output_targets['medianHouseValue'] /=1000
    return output_targets['medianHouseValue']



# 主方法
if __name__ == '__main__':
    # 读取数据 共20640条
    dataframe = pd.read_csv('data/cal_housing.csv')
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))

    # 训练集
    training_samples = preprocess_features(dataframe.head(15640))
    training_targets = preprocess_targets(dataframe.head(15640))

    # 验证集
    validation_samples = preprocess_features(dataframe.tail(5000))
    validation_targets = preprocess_targets(dataframe.tail(5000))

    # 训练
    linear_regressor = train_model(
                            training_samples=training_samples,
                            training_targets=training_targets,
                            validation_samples=validation_samples,
                            validation_targets=validation_targets,
                            learning_rate=0.00001,
                            batch_size=5,
                            steps=1000)

    plt.show()


    # correlation_dataframe = training_samples.copy()
    # correlation_dataframe['target']= training_targets
    # print(correlation_dataframe.corr())


