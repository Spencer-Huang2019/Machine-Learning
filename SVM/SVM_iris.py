from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

iris = load_iris()
# print(iris.data.shape)
X = iris.data[:,0:2]
y = iris.target
# print(X.shape)
# print(y)

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

# plot
x1_min, x1_max = X[:,0].min(), X[:,0].max()
x2_min, x2_max = X[:,1].min(), X[:,1].max()
x1,x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
grid_test = np.stack((x1.flat, x2.flat), axis=1)

matplotlib.rcParams['font.sans-serif']=['SimHei']

cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])

grid_hat = classifier.predict(grid_test)  # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本
plt.scatter(X_test[:, 0], X_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel('花萼长度', fontsize=13)
plt.ylabel('花萼宽度', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('鸢尾花SVM二特征分类')
plt.show()