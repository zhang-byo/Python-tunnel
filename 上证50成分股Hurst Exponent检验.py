#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import *
from numpy import *
import pandas as pd
import tushare as ts
from numpy.random import randn


get_sz50  = ts.get_sz50s()#tushare上获取上证50
symbol_dict = {}
code_list  = get_sz50['code']
name_list = get_sz50['name']
for  i in range(len(code_list )):#生成股票代码和股票名称字典
   symbol_dict[code_list[i]] = name_list[i]
print(symbol_dict)
#定义求赫斯特指数的函数
def hurst(ts):
   #创建时间间隔/滞后值的range
   lags = range(2,100)
   #计算滞后差分项变量的数组
   tau = [sqrt(std(subtract(ts[lag:],ts[:-lag]))) for lag in lags]
   #用线性拟合来估计赫斯特指数
   poly = polyfit(log(lags),log(tau),1)
   #从线性拟合输出返回赫斯特指数
   return poly[0]*2.0

start='2016-11-01'
end='2017-11-11'
hstcode_list = []
for code in code_list:
    df = ts.get_hist_data(code,start=start,end=end)
    close_df = df['close']
    hst = hurst(close_df)
    #print(adf)
    if hst < 0.2: #设置阈值为0.2
        hstcode_list.append(code)
        print(code,'是平稳的')
print(hstcode_list)#打印通过赫斯特检验后符合均值回归的股票列表
hst_dict = {}
for hstcode in hstcode_list:
    hst_dict[hstcode] = symbol_dict[hstcode]
print(hst_dict)
