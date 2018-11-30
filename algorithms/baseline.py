def myBaseline(data, trainLen):
    '''
    # 最近15分钟的数据作为15分钟(3个数据点)的预测值
    data(list):         所有的数据(包括训练数据和测试数据)
    trainLen(integer):  训练数据长度
    '''

    pred_data = []
    length = len(data)
    STEP = 3
    i = trainLen
    while i < length:
        x = sum(data[i-STEP:i]) / (STEP + 0.0)
        pred_data += min(STEP, length - i) * [x]
        i += min(STEP, length - i)

    return (data[trainLen:], pred_data)

# # test
# (x, y) = oldBaseline([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 6)
# print(len(x), x)
# print(len(y), y)