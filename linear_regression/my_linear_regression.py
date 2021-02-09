import numpy as np
import sympy as sp
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

class my_linear_regression(object):

    # calculate error
    def error(self, X_train, y_train):
        return y_train - np.dot(X_train, self.weight)

    # calculate gradient
    def gradient(self, X_train, y_train, i):
        return np.dot(self.error(X_train, y_train), (X_train[:, i].reshape(1, -1))).sum() / self.n_sample

    # define the training model
    def BGD_fit(self, X_train, y_train, learning_rate):

        self.learning_rate = learning_rate

        #  add xm=1 to X_train
        self.n_sample = X_train.shape[0]
        a = np.ones(self.n_sample)
        X_train = np.insert(X_train, 0, values=a, axis=1)
        n_feature = X_train.shape[1]

        #  initiate weight and gradient
        self.weight = np.ones((n_feature,1))
        gradient = np.zeros(n_feature)

        #  update weight
        for i in range(0, n_feature):
            diff = 1
            while (abs(diff) > 0.001):
                # get gradient for per weight
                gradient[i] = self.gradient(X_train, y_train, i)
                self.weight[i] = self.weight[i] + self.learning_rate * gradient[i]
                diff = gradient[i]

        # 4. return the optimal coef_ and intercept_
        self.coef_ = self.weight[1:]
        self.intercept_ = self.weight[0]
        return self.coef_, self.intercept_

    def SGD_fit(self, X_train, y_train, learning_rate):
        self.learning_rate = learning_rate

        #  add xm=1 to X_train
        self.n_sample = X_train.shape[0]
        a = np.ones(self.n_sample)
        X_train = np.insert(X_train, 0, values=a, axis=1)
        n_feature = X_train.shape[1]

        #  initiate weight and gradient
        self.weight = np.ones((n_feature, 1))
        gradient = np.zeros(n_feature)

        #  update weight
        for i in range(0, n_feature):
            for j in range(0, self.n_sample):
                gradient[i] = self.gradient(X_train[j, :].reshape(1, -1), y_train[j].reshape(1, -1), i)
                self.weight[i] = self.weight[i] + self.learning_rate * gradient[i]
                if gradient[i] < 0.001:
                    break

        # 4. return the optimal coef_ and intercept_
        self.coef_ = self.weight[1:]
        self.intercept_ = self.weight[0]
        return self.coef_, self.intercept_

    def predict(self, X_test):
        # 1. add xm = 1 to X_test
        n_sample = X_test.shape[0]
        a = np.ones(n_sample)
        X_test = np.insert(X_test, 0, values= a, axis=1)
        # 2. predict the result using trained coef_ and intercept_ and return the result matrix
        y_predict = np.dot(self.weight.T, X_test.T)
        return y_predict.T



if __name__ == "__main__":
    boston = load_boston()
    dim = 13
    X = boston.data
    y = boston.target

    # normalization: if omit this step, it will cause the explosion of gradient, as a result you will get nan weight
    Xx = StandardScaler()
    yy = StandardScaler()

    X = Xx.fit_transform(X.reshape(-1, dim))
    y = yy.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    linear_r = my_linear_regression()

    bgd = linear_r.BGD_fit(X_train.reshape(-1, X_train.shape[1]), y_train, learning_rate=0.001)
    bgd_predict = linear_r.predict(X_test)
    sgd = linear_r.SGD_fit(X_train.reshape(-1, X_train.shape[1]), y_train, learning_rate=0.001)
    sgd_predict = linear_r.predict(X_test)

    # print(linear_r.coef_)
    # print(linear_r.intercept_)
    print("mean squared error of BGD= ", mean_squared_error(y_test, bgd_predict))
    print("mean squared error of SGD= ", mean_squared_error(y_test, sgd_predict))

# figure when dimension = 1
#     plt.figure(figsize=(10, 6))
#     plt.title("Standard Gradient Descend")
#     plt.scatter(X_test, y_test)
#     plt.plot(X_test, linear_r.coef_*X_test + linear_r.intercept_, 'r')
#     plt.show()