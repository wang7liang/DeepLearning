import tensorflow as tf
from tensorflow.data import Dataset
import pandas as pd
import numpy as np
from sklearn import metrics
import math
import matplotlib.pyplot as plt



def input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # features = {key: value for key, value in dict(features).items()}
    features = {key: np.array(value) for key, value in dict(features).items()}
    # features = dict(features)

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(10000)
    inputs, labels = ds.make_one_shot_iterator().get_next()
    return inputs, labels


def train_model(dataframe, input_feature, learning_rate, steps, batch_size):
    features = dataframe[[input_feature]]
    targets = dataframe['target']

    feature_columns = [tf.feature_column.numeric_column(input_feature)]
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 1200.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=optimizer
    )
    linear_regressor.train(
        input_fn=lambda: input_fn(features, targets, batch_size),
        steps=steps
    )

    predictions = linear_regressor.predict(
        input_fn=lambda: input_fn(features, targets, shuffle=False, num_epochs=1)
    )
    predictions = np.array([item['predictions'][0] for item in predictions])

    mse = metrics.mean_squared_error(predictions,targets)
    rmse = math.sqrt(mse)
    print('mse: %0.3f, rmse: %0.3f' % (mse,rmse))

    samples = dataframe.sample(n=150)
    x_0 = samples[input_feature].min()
    x_1 = samples[input_feature].max()

    w = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    b = linear_regressor.get_variable_value('linear/linear_model/bias_weights')[0]
    y_0 = w*x_0+b
    y_1 = w*x_1+b

    plt.plot([x_0,x_1],[y_0,y_1],c='r')
    plt.scatter(samples[input_feature],samples['target'])
    # print(samples[input_feature])

    calibaration_data = pd.DataFrame()
    calibaration_data['predictions'] = pd.Series(predictions)
    calibaration_data['targets'] = pd.Series(targets)
    return calibaration_data


def preprocess_features(dataframe):
    selected_features = dataframe[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']]
    selected_features['new'] = (dataframe['ZN']).apply(lambda x: min(x,10))
    return selected_features


def preprocess_targets(dataframe):
    output_targets = pd.DataFrame()
    output_targets['target'] =dataframe['target']/1
    return output_targets


if __name__ == '__main__':
    # 一共506条数据
    dataframe = pd.read_csv('data/boston_data.csv')
    # 训练集
    training_samples = preprocess_features(dataframe.head(406))
    print(training_samples.describe())
    training_targets = preprocess_targets(dataframe.head(406))
    print(training_targets.describe())

    # 验证集
    validation_samples = preprocess_features(dataframe.tail(100))
    print(validation_samples.describe())
    validation_targets = preprocess_targets(dataframe.tail(100))
    print(validation_targets.describe())

    # calibaration_data = train_model(
    #     dataframe=dataframe,
    #     input_feature='ZN',
    #     learning_rate=0.00001,
    #     steps=10000,
    #     batch_size=10)

    # plt.figure()
    # plt.scatter(calibaration_data['predictions'], calibaration_data['targets'])

    # dataframe['ZN'].hist()
    # plt.show()

