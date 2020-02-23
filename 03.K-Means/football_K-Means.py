#!/usr/bin/env python3
#coding: utf-8 

from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
# 导入数据
data = pd.read_csv("data.csv", encoding='gbk')

train_x = data[["2019年国际排名", "2018世界杯", "2015亚洲杯"]]
df = pd.DataFrame(train_x)
# 归一化
mm = preprocessing.MinMaxScaler()
train_x = mm.fit_transform(train_x)
print("==============================")
print("==          分3类          ===")

kmeans = KMeans(n_clusters=3)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
result.rename({0:"聚类"}, axis=1, inplace=True)
print(result)

print("===========================================")
print("==     分5类（最小值最大值标准化）      ===")
kmeans = KMeans(n_clusters=5)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
result.rename({0:"聚类"}, axis=1, inplace=True)
print(result)

print("===========================================")
print("==     分5类（正太分布标准化）      =======")
ss = preprocessing.StandardScaler()
train_x = ss.fit_transform(train_x)
kmeans = KMeans(n_clusters=5)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
result.rename({0:"聚类"}, axis=1, inplace=True)
print(result)
