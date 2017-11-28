#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import tushare as ts
import numpy as np
import statsmodels.tsa.stattools as sts


get_sz50  = ts.get_sz50s()#tushare上获取上证50
symbol_dict = {}
code_list  = get_sz50['code']
name_list = get_sz50['name']
for  i in range(len(code_list )):#生成股票代码和股票名称字典
   symbol_dict[code_list[i]] = name_list[i]
print(symbol_dict)

start = '2016-11-01'
end = '2017-11-11'
adfcode_list = []
for code in code_list:
    df = ts.get_hist_data(code, start=start, end=end)
    close_df = df['close']
    adf = sts.adfuller(close_df, 1)
    # print(adf)
    if adf[0] < adf[4]['5%']:
        adfcode_list.append(code)
        print(code, '是平稳的')

print(adfcode_list)  # 打印符合均值回归（即是平稳的）的股票列表

adf_dict = {}
for adfcode in adfcode_list:
    adf_dict[adfcode] = symbol_dict[adfcode]
print(adf_dict)#打印符合均值回归的股票名称和代码
