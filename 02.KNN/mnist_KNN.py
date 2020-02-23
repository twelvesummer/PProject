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
#print(data[0])#��ʾ��һ��ͼ������
#print(target[0])#��ʾ��һ��ͼ�Ľ��
#print(digits.images[0])#��ʾ��һ��ͼ��8*8���ؽ������ά����
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
#plt.imshow(digits.images[0])#M*N      ��ʱ�������Ϊ�����ͣ�����ֵΪ������ĻҶ�
#plt.show()
#plt.savefig('mnits.01.png')

# ѵ�����Ͳ��Լ��ָ�
train_x, test_x, train_y, test_y = train_test_split(data,target, test_size=0.25, random_state=22)
# train_data �����ֵ�����������
# train_target �����ֵ�������ǩ
# test_size ����Ǹ���������ʾ����ռ�ȣ������������������������
# random_state ����������ӣ�����������ı�ţ������Ҫ�ظ������ʱ��֤�õ�һ��һ���������������1����������һ��������µõ������������һ���ģ���0���߲�����ÿ�ζ���һ����������ͬ����ʹʵ����ͬҲ������ͬ���������
# z-score��׼��
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)
# fit_transform��fit��transform����������ִ��һ�Ρ�����ss�ǽ�����fit��ϵġ�ֻ����fit���֮�󣬲��ܽ���transform�ڽ���test��ʱ�������Ѿ���train��ʱ��fit���ˣ�����ֱ��transform���ɡ����⣬�������û��fit��ֱ�ӽ���transform�ᱨ����Ϊ��Ҫ��fit��ϣ��ſ��Խ���transform��

# KNNѵ������Ԥ��
knn = KNeighborsClassifier()
time_start = time.time()
knn.fit(train_x, train_y)
predict_y = knn.predict(test_x)
time_stop = time.time()
# ׼ȷ�Է���
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
# ����MIn-Max�淶��
mm = MinMaxScaler()
train_x = mm.fit_transform(train_x)
test_x = mm.transform(test_x)
# Native Bayes ���벻���и��������Բ����ñ�׼��̫�ֲ�����һ��
mnb = MultinomialNB()
mnb.fit(train_x, train_y)
predict_mnb = mnb.predict(test_x)
print("NB accuracy: %.4lf" % accuracy_score(test_y, predict_mnb))

# CART����
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(train_x, train_y)
predict_clf = clf.predict(test_x)
print("CART accuracy:%4lf" % accuracy_score(test_y, predict_clf))
