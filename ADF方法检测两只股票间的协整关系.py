#--*--coding:utf-8 --*--
"""
   author:zhang-byo
   Email:137960572@qq.com

   用ADF方法检测两支股票的协整关系

   首先用OLS方法就两只股票进行线性拟合，然后对拟合后的残差做ADF检测
   判断残差的平稳性，从而判别两只股票的协整关系

   对代码进行了详细标注
"""
import matplotlib.pyplot as plt
import pandas as pd
import tushare as tus
import statsmodels.tsa.stattools as ts #用里面的ADF检测模型
import statsmodels.api as sm  #用里面的OLS（常规最小二乘）模型

def plot_price_series(df, ts1, ts2):
    fig, ax = plt.subplots() #创建图例，返回ax数组
    ax.plot(df.index, df[ts1], label=ts1)#画ts1图,x为日期，y为价格
    ax.plot(df.index, df[ts2], label=ts2)#画ts2图
    ax.grid(True)#显示网格
    fig.autofmt_xdate()# 日期的排列根据图像的大小自适应
    plt.xlabel('Month/Year')
    plt.ylabel('Price (¥)')
    plt.title('%s and %s Daily Prices' % (ts1, ts2))
    plt.legend()#显示图例
    plt.show()

def plot_scatter_series(df, ts1, ts2):
    plt.xlabel('%s Price (¥)' % ts1)
    plt.ylabel('%s Price (¥)' % ts2)
    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
    plt.scatter(df[ts1], df[ts2])#散点图
    plt.show()

def plot_residuals(df):#画残差
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel('Month/Year')
    plt.ylabel('Price (¥)')
    plt.title('Residual Plot')
    plt.legend()
    plt.plot(df["res"])
    plt.show()


if __name__ == "__main__":
    start = '2017-01-01'
    end = '2017-03-01'
    arex = tus.get_hist_data('600847', start=start, end=end)#tushare上获取数据
    wll = tus.get_hist_data('600848', start=start, end=end)
    df = pd.DataFrame(index=arex.index)
    df["stock_a"] = arex["close"] #取收盘价
    df["stock_b"] = wll["close"]

    plot_price_series(df, "stock_a", "stock_b")#画出两个时间序列图（曲线图）

    plot_scatter_series(df, "stock_a", "stock_b")#画散点图

    X = df["stock_a"]#待OLS的X集合
    # X = sm.add_constant(X)
    Y = df['stock_b']
    # 计算 optimal hedge ratio（最优套期保值率） beta
    model = sm.OLS(Y, X)
    res = model.fit()
    beta = res.params

    # 计算两股票线性组合的残差residuals
    # df["res"] = df["WLL"] - df["AREX"]
    df["res"] = df["stock_b"] - df["stock_a"].apply(lambda x: beta.values * x)
    plot_residuals(df)#画残差图
    #对残差进行ADF检验并输出结果
    cadf = ts.adfuller(df["res"])
    print(cadf)
    #检测结果第一项为test statistic 大于5% 临界值 即可认为两个股票的协整关系为非平稳的

