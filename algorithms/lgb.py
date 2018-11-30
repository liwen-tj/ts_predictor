import lightgbm as lgb
import numpy as np


def getFeature(data, trainLen, FEATURE_NUM):
    # 转换成X(特征, features) + Y(labels)的形式
    n = len(data) - FEATURE_NUM
    features = []
    labels = []
    for i in range(n):
        features.append(data[i:i+FEATURE_NUM])
        labels.append(data[i+FEATURE_NUM])

    trainX = np.array(features[:trainLen-FEATURE_NUM])
    trainY = np.array(labels[:trainLen-FEATURE_NUM])
    testX = np.array(features[trainLen-FEATURE_NUM:])
    testY = np.array(labels[trainLen-FEATURE_NUM:])

    return (trainX, trainY, testX, testY)


def myLightgbm(data, trainLen):
    '''
    data(list):         所有的数据(包括训练数据和测试数据)
    trainLen(integer):  训练数据长度
    '''
    FEATURE_NUM = 120 # 10个小时的数据
    (trainX, trainY, testX, testY) = getFeature(data, trainLen, FEATURE_NUM)

    # 模型
    params = {
        'learning_rate': 0.1,
        'objective': 'regression_l1',
        'max_depth': 4,
        'num_leaves': 4,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 20,
        'metric': 'mape',
        'seed': 42
    }

    train_data = lgb.Dataset(trainX, label=trainY)
    bst = lgb.train(params, train_data, num_boost_round=500)
    ypred = bst.predict(testX)

    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
    
    print(type(testY), type(ypred))
    print(testY.shape, ypred.shape)
    return (testY, ypred)
