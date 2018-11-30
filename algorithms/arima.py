from statsmodels.tsa.arima_model import ARIMA


def myArima(data, trainLen):
    '''
    data(list):         所有的数据(包括训练数据和测试数据)
    trainLen(integer):  训练数据长度
    '''
    train_data = data[trainLen-600:trainLen] # 初始训练数据
    test_data = data[trainLen:] # 测试数据
    pred_data = [] # 预测数据(应该与测试数据test_data长度一致)

    testLen = test_data.__len__()
    STEP = 3 # 每次同时预测STEP个数据点
    i = 0
    while i < testLen:
        # 搭建模型
        model = ARIMA(train_data, order=(5, 1, 0)) # order=(p, q, d)分别是AR, I, MA的参数
        model_fit = model.fit()
        if testLen - i < 2 * STEP: # 最后剩下的少量数据一次性预测
            STEP = testLen - i
        # forecast函数默认执行一步预测操作，可以通过设置steps进行
        # 当steps等于1时，output是一个数字;当steps大于1时，output是一个长度为steps的list
        output = list(model_fit.forecast(steps=STEP)[0])
        pred_data += output
        train_data = train_data[STEP:] + test_data[i:i+STEP] # 下次再预测就要更新数据了
        i += STEP
        print("==================##########", i, "#########==================", '\n\n\n')
    
    return (test_data, pred_data)

