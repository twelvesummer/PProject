import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
train_data['Fare'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Fare'].fillna(train_data['Age'].mean(),inplace=True)

train_data['Embarked'].fillna('S',inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
test_features=dvec.transform(test_features.to_dict(orient='record'))

ada = AdaBoostClassifier()
ada.fit(train_features, train_labels)
print("ada train precision = ",ada.score(train_features, train_labels))
print("ada 10k precison = ", np.mean(cross_val_score(ada,train_features,train_labels,cv=10)))
clf=DecisionTreeClassifier(criterion='entropy')
clf.fit(train_features,train_labels)
print("clf train precision = ", clf.score(train_features, train_labels))
print("clf 10k precision = ", np.mean(cross_val_score(clf,train_features,train_labels,cv=10)))
