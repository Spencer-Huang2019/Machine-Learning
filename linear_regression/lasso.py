from sklearn import datasets
from sklearn.linear_model import Lasso, LassoCV
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

#1. import the boston dataset
boston = datasets.load_boston()

df = pd.DataFrame(boston.data)
X = boston.data
y = boston.target  # get labels

# print(x)
# print(y)

#2. split the dataset into training set and testing set
X_train0, X_test0, y_train0, y_test0 = train_test_split(X, y, test_size= 0.3)

# print(X_train.shape)
# print(y_train.shape)

#3. normalization data

ss_X = StandardScaler()
ss_y = StandardScaler()

X_train = ss_X.fit_transform(X_train0)
X_test = ss_X.fit_transform(X_test0)

y_train = ss_y.fit_transform(y_train0.reshape(-1, 1))
y_test = ss_y.fit_transform(y_test0.reshape(-1, 1))

# get the optimal alpha
alphaL = LassoCV(cv=10, alphas=list(np.arange(0.03,0.6,0.001))).fit(X_train, y_train)
print("alpha= ", alphaL.alpha_)

#4. Training & testing
lassoR = Lasso(max_iter=10000, alpha=alphaL.alpha_)

# fit the data with training samples
lassoR_predict = lassoR.fit(X_train, y_train).predict(X_test)

print(lassoR.coef_)

#6. evaluation
print('mean_square_error: %.2f' %mean_squared_error(y_test, lassoR_predict))
# print('Coefficient of determination: %.2f' %r2_score(y_test, lassoR_predict))
print('Mean absolute error of lasso: %.2f' %mean_absolute_error(y_test, lassoR_predict))


# # plot the figure
# plt.figure(figsize=(10, 50))
# plt.xlim([0, 50])
# plt.plot(range(len(y_test)), y_test, 'r', label='y_test')
# plt.plot(range(len(lassoR_predict)), lassoR_predict, 'p--', label='lassoR_predict')
#
# plt.legend()
#
# plt.show()