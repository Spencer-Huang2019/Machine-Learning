from sklearn import datasets
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#import the boston dataset
boston = datasets.load_boston()

df = pd.DataFrame(boston.data)
x = boston.data
y = boston.target  # get labels

# print(x)
# print(y)

# split the dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.3)

# print(X_train.shape)
# print(y_train.shape)

# normalization data
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()

X_train = ss_X.fit_transform(X_train.reshape(-1, 13))
X_test = ss_X.transform(X_test.reshape(-1, 13))

y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# instantiate a LinearRegression method
sgdLR = SGDRegressor()
lineR = LinearRegression()
# fit the data with training samples
sgdLR.fit(X_train, y_train)  #  "-1" means don't care split to how many lines, all I want is splitting to 1 column
lineR.fit(X_train, y_train)

# testing
predictionsS = sgdLR.predict(X_test)
predictionsL = lineR.predict(X_test)

print("mean squared error of Standard= ", mean_squared_error(predictionsL, y_test))
print("mean squared error of SGD= ", mean_squared_error(predictionsS, y_test))

# scoreSTrain = sgdLR.score(X_train, y_train)
# scoreSTest = sgdLR.score(X_test, y_test)
# print('Score of sgdLR train: %.5f' % scoreSTrain)
# print('Score of sgdLR test: %.5f' % scoreSTest)
#
# scoreLTrain = lineR.score(X_train, y_train)
# scoreLTest = lineR.score(X_test, y_test)
# print('Score of lineR train: %.5f' % scoreLTrain)
# print('Score of lineR test: %.5f' % scoreLTest)
#
# # plot the figure
# plt.figure(figsize=(10, 50))
# plt.xlim([0, 50])
# plt.plot(range(len(y_test)), y_test, 'r', label='y_test')
# plt.plot(range(len(predictionsS)), predictionsS, 'g--', label='SGD prediction')
# plt.plot(range(len(predictionsL)), predictionsL, 'b--', label='standard prediction')
# plt.legend()
#
# plt.show()