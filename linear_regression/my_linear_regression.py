import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

class my_linear_regression(object):

    def forward(self, x):
        return np.dot(x, self.weight)

    def cost(self, X, Y):
        cost = 0
        for x, y in zip(X, Y):
            y_pred = self.forward(x)
            cost += (y_pred - y) ** 2
        return cost / len(X)

    def gradient(self, X, Y):
        n_feature = X_train.shape[1]
        grad = np.zeros(n_feature + 1)

        for x, y in zip(X, Y):
            grad += 2 * x * (self.forward(x) - y)
        return grad / len(Y)

    # calculate gradient
    def gradientR(self, y_pred, y_train, x):
        return (y_pred - y_train).sum()*x/self.n_sample

    # define the training model

    def BGD_fit_1(self, X_train, y_train, learning_rate):

        self.learning_rate = learning_rate

        #  add xm=1 to X_train
        self.n_sample = X_train.shape[0]
        a = np.ones(self.n_sample)
        X_train = np.insert(X_train, 0, values=a, axis=1)
        n_feature = X_train.shape[1]

        #  initiate weight and gradient
        self.weight = np.ones((n_feature,1))
        weight_temp = np.ones((n_feature, 1))
        gradient = np.zeros(n_feature)

        #  update weight
        while(gradient.sum() > 0.001):
            y_pred = np.dot(X_train, self.weight)
            for i in range(0, n_feature):
                # get gradient for per weight
                gradient[i] = -self.gradientR(y_pred, y_train, i)
                self.weight[i] = self.weight[i] - self.learning_rate * gradient[i]

        # 4. return the optimal coef_ and intercept_
        self.coef_ = self.weight[1:]
        self.intercept_ = self.weight[0]
        return self.coef_, self.intercept_

    def BGD_fit(self, X_train, y_train, learning_rate):

        self.learning_rate = learning_rate

        self.n_sample = X_train.shape[0]
        a = np.ones(self.n_sample)
        X_train = np.insert(X_train, 0, values=a, axis=1)
        n_feature = X_train.shape[1]

        #  initiate weight and gradient
        self.weight = np.ones((n_feature, 1))
        gradient = np.zeros(n_feature)

        #  update weight
        while(gradient.sum() > 0.001):
            y_pred = np.dot(X_train, self.weight)
            for i in range(0, n_feature):
                for j in range(0, self.n_sample):
                    # get gradient for per weight
                    gradient[i] = -self.gradientR(y_pred, y_train, X_train[j][i])
                    self.weight[i] = self.weight[i] - self.learning_rate * gradient[i]

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
            y_pred = np.dot(X_train, self.weight)
            for j in range(0, self.n_sample):
                gradient[i] = -self.gradientR(y_pred[j], y_train[j], i)
                self.weight[i] = self.weight[i] - self.learning_rate * gradient[i]
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
        y_predict = np.dot(X_test, self.weight)
        return y_predict

if __name__ == "__main__":

    boston = load_boston()
    dim = 2
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
    # sgd = linear_r.SGD_fit(X_train.reshape(-1, X_train.shape[1]), y_train, learning_rate=0.001)
    # sgd_predict = linear_r.predict(X_test)

    # print(linear_r.coef_)
    # print(linear_r.intercept_)
    print("mean squared error of BGD= ", mean_squared_error(y_test, bgd_predict))
    # print("mean squared error of SGD= ", mean_squared_error(y_test, sgd_predict))


# figure when dimension = 1
#     plt.figure(figsize=(10, 6))
#     plt.title("Standard Gradient Descend")
#     plt.scatter(X_test, y_test)
#     plt.plot(X_test, linear_r.coef_*X_test + linear_r.intercept_, 'r')
#     plt.show()