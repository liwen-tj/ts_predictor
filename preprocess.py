import pandas as pd


def readCpuUsage(filename):
    ''' 读取cpu利用率，并填充缺失值，返回list类型的数据 '''
    # 读取数据
    raw_data = pd.read_csv(filename, sep=';\t', engine='python')
    cpu_usage_raw = raw_data['CPU usage [%]']
    timestamp = raw_data['Timestamp [ms]']

    # 填充缺失值
    cpu_usage = [cpu_usage_raw[0]] # 先放第一个数据进去
    records_num = raw_data.shape[0]
    for i in range(1, records_num):
        interval = timestamp[i] - timestamp[i-1]
        if(200 < interval < 400):
            cpu_usage.append(cpu_usage_raw[i])
        elif(interval >= 400):
            n = int(interval / 300)
            cpu_usage += [cpu_usage_raw[i] for _ in range(n)]

    # 返回预处理完成的数据
    return cpu_usage # list
