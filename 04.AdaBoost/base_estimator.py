import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

X,y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
train_x, train_y = X[2000:], y[2000:]#
test_x, test_y = X[:2000],y[0:2000]

dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(train_x, train_y)
dt_stump_err = 1 - dt_stump.score(test_x, test_y)

ada = AdaBoostClassifier(base_estimator=dt_stump, n_estimators=200)
ada.fit(train_x, train_y)

dt = DecisionTreeClassifier()
dt.fit(train_x, train_y)
dt_err = 1.0 - dt.score(test_x, test_y)

n_estimators = 200
fig = plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']
ax = fig.add_subplot(111)
ax.plot([1, n_estimators], [dt_stump_err]*2, 'k-', label=u'dt_stump err')
ax.plot([1, n_estimators], [dt_err]*2, 'k--', label='decisionTree err')
ada_err = np.zeros((n_estimators,))
for i,pred_y in enumerate(ada.staged_predict(test_x)):
    ada_err[i] = zero_one_loss(pred_y, test_y)
ax.plot(np.arange(n_estimators)+1, ada_err, label='AdaBoost Test err', color='orange')
ax.set_xlabel('estimator')
ax.set_ylabel('err')
leg=ax.legend(loc='upper right',fancybox=True)
plt.savefig('base_estimator.png')
plt.close()
