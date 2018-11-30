import imp
from sklearn.metrics import mean_squared_error, mean_absolute_error
from preprocess import readCpuUsage
from algorithms.baseline import myBaseline
from algorithms.arima import myArima
from algorithms.lgb import myLightgbm
from algorithms.lstm import myLstm
import time


if __name__ == '__main__':
    start = time.time()
    cpuUsage = readCpuUsage('../fastStorage/2013-8/272.csv')
    (y_true, y_pred) = myLstm(cpuUsage, 7000)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    end = time.time()
    print('len =', len(y_pred))
    print('time cost =', end - start, 's')
    print('mse =', mse)
    print('rmse =', mse**0.5)
    print('mae =', mae)
    print('data_avg =', sum(cpuUsage[7000:]) / (len(cpuUsage[7000:]) + 0.0))
    