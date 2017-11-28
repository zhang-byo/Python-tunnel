#--*--coding:utf-8 --*--
"""
   author:zhang-byo
   Email:137960572@qq.com

   从tushare网上下载数据并进行清洗处理后保存到本地

   1、在指标中添加CCI
   2、对标签值y进行了设定

"""
import numpy as np
import pandas as pd
import tushare as ts

#网上获取数据
start = '2016-01-01'
end = '2017-05-01'
code = '601390'
stock_a = ts.get_hist_data(code, start=start, end=end)
stock_a = stock_a.sort_index(axis=0,ascending=True)#对数据按照日期进行排序，有利于计算CCI值

# 定义CCI函数
def CCI(data, ndays):
    TP = (data['high'] + data['low'] + data['close']) / 3
    CCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)), name='CCI')
    data = data.join(CCI)
    return data

if __name__ == "__main__":
    n = 20#时间窗格设为20天
    NIFTY_CCI = CCI(stock_a, n)# 添加CCI指标作为特征值
    stock_a = NIFTY_CCI.dropna()  # 去掉空值所在行
    # 以close值作为y值，并做相应调整（把t时间的y值设为t+1时刻的y值）
    stock_a["y"] = stock_a["close"]
    print(stock_a.head())
    for i in range(len(stock_a["y"])):
        if i == len(stock_a["y"]) - 1:
            stock_a["y"][i] = None
        else:
            stock_a["y"][i] = stock_a["close"][i + 1]
    stock_df = stock_a.dropna()#去掉最后一行空值
    stock_df.to_csv('./%s.csv' % code)#保存到本地
