#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler

digits = load_digits()
target = digits.target
data = digits.data# numpy.narray
#print(data.shape)
#print(data[0])#显示第一张图的数据
#print(target[0])#显示第一张图的结果
#print(digits.images[0])#显示第一张图的8*8像素结果，二维数组
'''
0
[[ 0.  0.  5. 13.  9.  1.  0.  0.]
 [ 0.  0. 13. 15. 10. 15.  5.  0.]
 [ 0.  3. 15.  2.  0. 11.  8.  0.]
 [ 0.  4. 12.  0.  0.  8.  8.  0.]
 [ 0.  5.  8.  0.  0.  9.  8.  0.]
 [ 0.  4. 11.  0.  1. 12.  7.  0.]
 [ 0.  2. 14.  5. 10. 12.  0.  0.]
 [ 0.  0.  6. 13. 10.  0.  0.  0.]]
'''
#plt.gray()
#plt.imshow(digits.images[0])#M*N      此时数组必须为浮点型，其中值为该坐标的灰度
#plt.show()
#plt.savefig('mnits.01.png')

# 训练集和测试集分割
train_x, test_x, train_y, test_y = train_test_split(data,target, test_size=0.25, random_state=22)
# train_data 被划分的样本特征集
# train_target 被划分的样本标签
# test_size 如果是浮点数，表示样本占比，如果是整数代表样本的数量
# random_state 随机数的种子，该组随机数的编号，如果需要重复试验的时候保证得到一组一样的随机数，如填1，其他参数一样的情况下得到的随机数组是一样的，填0或者不填则每次都不一样，种子相同，即使实例不同也产生相同的随机数。
# z-score标准化
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)
# fit_transform是fit和transform两个函数都执行一次。所以ss是进行了fit拟合的。只有在fit拟合之后，才能进行transform在进行test的时候，我们已经在train的时候fit过了，所以直接transform即可。另外，如果我们没有fit，直接进行transform会报错，因为需要先fit拟合，才可以进行transform。

# KNN训练集和预测
knn = KNeighborsClassifier()
time_start = time.time()
knn.fit(train_x, train_y)
predict_y = knn.predict(test_x)
time_stop = time.time()
# 准确性分析
print("KNN(k=5) accuracy: %.4lf\ntime:%.4f" % (accuracy_score(test_y, predict_y), time_stop - time_start))
knn = KNeighborsClassifier(n_neighbors=200)
time_start = time.time()
knn.fit(train_x, train_y)
predict_y = knn.predict(test_x)
time_stop = time.time()
print("KNN(k=200) accuracy: %.4lf\ntime:%.4f" % (accuracy_score(test_y, predict_y), time_stop - time_start))
# SVM
model = svm.SVC()
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print("SVM accuracy: %.4lf" % accuracy_score(test_y, prediction))
# 采用MIn-Max规范化
mm = MinMaxScaler()
train_x = mm.fit_transform(train_x)
test_x = mm.transform(test_x)
# Native Bayes 传入不能有负数，所以不能用标准正太分布来均一化
mnb = MultinomialNB()
mnb.fit(train_x, train_y)
predict_mnb = mnb.predict(test_x)
print("NB accuracy: %.4lf" % accuracy_score(test_y, predict_mnb))

# CART分类
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(train_x, train_y)
predict_clf = clf.predict(test_x)
print("CART accuracy:%4lf" % accuracy_score(test_y, predict_clf))
