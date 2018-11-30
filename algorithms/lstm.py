import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def createDataset(data, trainLen, look_back=3):
    dataX, dataY = [], []
    n = len(data) - look_back
    for i in range(n):
        dataX.append(data[i:i+look_back])
        dataY.append(data[i+look_back])
    
    t = trainLen - look_back
    (_trainX, _testX) = (np.array(dataX[:t]), np.array(dataX[t:]))
    (trainX, testX) = (np.reshape(_trainX, (_trainX.shape[0], look_back, 1)), np.reshape(_testX, (_testX.shape[0], look_back, 1)))
    
    return (trainX, np.array(dataY[:t]), testX, np.array(dataY[t:]))


def myLstm(data, trainLen):
    (trainX, trainY, testX, testY) = createDataset(data, trainLen)
    # 模型搭建
    model = Sequential()
    model.add(LSTM(5, input_shape=(3, 1)))
    model.add(Dense(1))

    # 模型训练
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=50, batch_size=10, shuffle=True)

    # 预测
    ys = model.predict(testX)
    preds = np.reshape(ys, (ys.shape[0], ))
    return (testY, preds)

