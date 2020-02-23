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
# ���ݼ���
data = pd.read_csv("data.csv")
## ���ñ��������ʾ�У�None����ȫ��ʾ����
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
# ������ϴ
## ��ID��ɾ��
data.drop('id', axis=1, inplace=True)
## ��diagnosis��M,BתΪ1��0
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
## ������ֵ��Ϊ3�飬ע���ʱid���Ѿ�ɾ��
feature_mean = data.columns[1:11]
feature_se = data.columns[11:21]
feature_worst = data.columns[21:31]
# ���ݿ��ӻ�
sns.countplot(data['diagnosis'],label='Count')
plt.savefig('diagnosis_countplot.png')
plt.show()
# ������ͼ����feature_mean�ֶ�֮��������
corr = data[feature_mean].corr()
plt.figure(figsize=(14,14))
# annot��ʾÿ�����������
sns.heatmap(corr, annot=True)
plt.savefig('feature_mean_corr.png')
plt.show()
# radius_mean��perimeter_mean��area_mean������Դ�compactness_mean��daconcavity_mean��concave point_mean����Դ󣬿��Էֱ����������ѡ��һ�����������ԡ�
feature_mean_2 = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean']
# ѵ�����Ͳ��Լ�
train, test = train_test_split(data, test_size= 0.3)
train_X = train[feature_mean_2]
train_y = train['diagnosis']
test_X = test[feature_mean_2]
test_y = test['diagnosis']
# ���ݹ�һ��
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.fit_transform(test_X)
# ѵ����Ԥ�⣬Ĭ���Ǹ�˹����
model = svm.SVC()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print("accuracy: ", metrics.accuracy_score(prediction, test_y))
