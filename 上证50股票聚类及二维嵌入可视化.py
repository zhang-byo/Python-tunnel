#--*--coding:utf-8 --*--
"""
Author:zhang-byo
Email:137960572@qq.com

借鉴sklearn官方例子，在tushare网上获取股票信息。
对上证50成分股进行聚类及二维嵌入可视化
并对程序进行了详细的标注

"""
from __future__ import print_function
import pandas as pd
import tushare as ts
import numpy as np
import os
import sys
reload(sys)
sys.setdefaultencoding('Cp1252')    #设置中文字体，解决显示不了中文问题
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  #设置中文字体，解决plt显示不了中文问题
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold

#开始从tushare网上获取上证50股票代码和名称
get_sz50  = ts.get_sz50s()
symbol_dict = {}
code_list  = get_sz50['code']
name_list = get_sz50['name']
for  i in range(len(code_list )):
   symbol_dict[code_list[i]] = name_list[i]#.decode('utf-8', 'ignore')，报错时加上这句

#小批量测试用
#code_list = [ '000792','601021','603993','600111','603160','002153','600115','002124','600637','600549']
#symbol_dict = {'000792':'one'.decode('utf-8', 'ignore') ,'601021':'two','603993':'three', '600111':'four','603160':'five','002153':'six','600115':'seven','002124':'eight','600637':'nine','600549':'ten'}

#指定开始时间和截止时间，在tushare上获取股票数据，选取开盘价和收盘价之差作为特征。对数据进行处理后保存为csv文件
start='2015-11-01'
end='2017-11-01'
cl_op_df = pd.DataFrame()
for code in code_list:
    df = ts.get_hist_data(code,start=start,end=end)
    close_df = df['close']
    open_df = df['open']
    df1 = close_df - open_df
    df1.name = code
    cl_op_df = cl_op_df.append(df1)
cl_op_df = cl_op_df
cl_op_df = cl_op_df.fillna(cl_op_df.mean())
cl_op_df.to_csv('./test.csv')

#symbols 为股票代码，name 为股票中文名称
symbols, names = np.array(sorted(symbol_dict.items())).T
# 根据协方差学习一个图形结构。可生成协方差矩阵，进而为下文求得偏相关系数矩阵做依据
edge_model = covariance.GraphLassoCV()
#标准化训练数据，在结构发现上，用相关系数比协方差更有效
X = cl_op_df.copy().T
X = X/X.std(axis=0)
edge_model.fit(X)
ed = edge_model.fit(X)
#用affinity propagation方法进行聚类
_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))#含条件语句的join 函数

#多维矩阵转为二维矩阵以便显示坐标系中，在二维空间中找到节点的最佳位置
# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=10)
#多维数组转变为二维数组embedding
embedding = node_position_model.fit_transform(X.T).T
# 生成画布和坐标范围
plt.figure(1, facecolor='w', figsize=(10, 8))  # 生成画布
plt.clf()  # 可要可不要
ax = plt.axes([0., 0., 1., 1.])  # 画轴心

# 绘偏相关系数图
partial_correlations = edge_model.precision_.copy()  # 协方差矩阵
# d为一维数组
# np.sqrt#平方根
# np.diag函数输出协方差矩阵的对角线（为一维数组）
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d  # 偏相关系数矩阵中间量
# d[:, np.newaxis]一维数组的转置方法，不能直接用.T方法
partial_correlations *= d[:, np.newaxis]  # 最终的偏相关系数矩阵,矩阵对角线元素为1
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)  # 返回满足条件的布尔矩阵
# np.triu(partial_correlations, k=1)对角线及对角线以下元素置为零


#用相关系数画点
plt.scatter(embedding[0], embedding[1], s=20 * d ** 2, c=labels,
            cmap=plt.cm.spectral)  # 画散点图,x,y 坐标为embedding[0],embedding[1];
# s:点的大小（一维数组）随d（1/标准差，意味着标准差越大，点的形状越小）变化，c：颜色（一维数组）随坐标变化

#绘制边界线（即两股票点的连接线）
start_idx, end_idx = np.where(non_zero)  # 只有条件non_zero，所以返回non_zero.nonzero()，即返回有两个元素的元组，
# 第一个元素为非零的数在O轴即竖轴的下标，第二个元素为非零的数在1轴即横轴的下标

# a sequence of (*line0*, *line1*, *line2*), where::linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
#embedding为二维数组

values = np.abs(partial_correlations[non_zero])  # 用non_zero遮罩后的15个元素的数组
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))  # zorder:调整层次，cmap:colormap

lc.set_array(values)
lc.set_linewidths(6 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate( zip(names, labels, embedding.T)):
    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name,size=5,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             #bbox=dict(facecolor='w',edgecolor=plt.cm.spectral(label / float(n_labels)),alpha=.6)
             )
plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(), )
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())
plt.savefig('./total.png', format='png')
plt.show()
