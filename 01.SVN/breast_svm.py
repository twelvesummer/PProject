#!/usr/bin/env python3
#-*- coding:utf-8 -*-
__author__ = 'guofengming'
__version__ = '1.0'
'''



'''

import pandas as pd
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
# 数据加载
data = pd.read_csv("data.csv")
## 设置表格的最大显示列，None是完全显示出来
pd.set_option('display.max_columns',None)
#print(data.describe())
#print(data.columns)
'''
Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst'],
      dtype='object')
'''
# 数据清洗
## 将ID列删除
data.drop('id', axis=1, inplace=True)
## 将diagnosis的M,B转为1，0
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
## 将特征值分为3组，注意此时id列已经删除
feature_mean = data.columns[1:11]
feature_se = data.columns[11:21]
feature_worst = data.columns[21:31]
# 数据可视化
sns.countplot(data['diagnosis'],label='Count')
plt.savefig('diagnosis_countplot.png')
plt.show()
# 用热力图呈现feature_mean字段之间的相关性
corr = data[feature_mean].corr()
plt.figure(figsize=(14,14))
# annot显示每个方格的数据
sns.heatmap(corr, annot=True)
plt.savefig('feature_mean_corr.png')
plt.show()
# radius_mean、perimeter_mean、area_mean的相关性大，compactness_mean、daconcavity_mean、concave point_mean相关性大，可以分别从这两类中选择一个代表性属性。
feature_mean_2 = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean']
# 训练集和测试集
train, test = train_test_split(data, test_size= 0.3)
train_X = train[feature_mean_2]
train_y = train['diagnosis']
test_X = test[feature_mean_2]
test_y = test['diagnosis']
# 数据归一化
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.fit_transform(test_X)
# 训练和预测，默认是高斯函数
model = svm.SVC()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print("accuracy: ", metrics.accuracy_score(prediction, test_y))
