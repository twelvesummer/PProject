#-*- coding:utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
X,y = load_boston(return_X_y=True)
#print(data.data)#data.ndarray
#print(X.shape)
train_x, test_x, train_y, test_y= train_test_split(X, y, test_size=0.25, random_state=33)
regressor = AdaBoostRegressor()
regressor.fit(train_x, train_y)
pred_y = regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
#print(u"pred_y", pred_y)
print(u"AdaBoost mse = ", round(mse,2))

dec_regressor = DecisionTreeRegressor()
dec_regressor.fit(train_x, train_y)
pred_y = dec_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("decison mse = ", round(mse,2))

knn_regressor = KNeighborsRegressor()
knn_regressor.fit(train_x, train_y)
pred_y = knn_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("KNN mse = ", round(mse,2))
