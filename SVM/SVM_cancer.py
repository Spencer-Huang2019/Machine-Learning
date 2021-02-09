from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn import svm

from matplotlib import pyplot as plt
import matplotlib
import numpy as np

cancer = load_breast_cancer()
# print(iris.data.shape)
X = cancer.data
y = cancer.target
# print(X.shape)
# print(y)

# dimension reduction


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)

# normalization
# ssX = StandardScaler()
# ssY = StandardScaler()
#
# X_train = ssX.fit_transform(X_train)
# X_test = ssX.fit_transform(X_test)
#
# y_train = ssY.fit_transform(y_train)
# y_test = ssY.fit_transform(y_test)

# Training by SVM
# {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
classifier = svm.SVC(C=2, kernel='rbf',gamma=10,decision_function_shape='ovo')
classifier.fit(X_train, y_train.ravel()) # ravel: when degrading dimemtion, row order is the first

# estimate
print("Training set: ", classifier.score(X_train, y_train.reshape(-1,1)))
print("Testing set: ", classifier.score(X_test, y_test.reshape(-1,1)))

