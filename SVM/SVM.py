from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=1, n_redundant=0,
                                   n_repeated=0, n_clusters_per_class=1, class_sep=2.0)
plt.scatter(X[:,0],X[:,1],c=y)

# svm设置：
clf0 = svm.SVC(kernel='linear')
clf0.fit(X, y)

# 获取w
w0 = clf0.coef_[0]
print('w', w0)
a0 = -w0[0] / w0[1]  # 斜率
# 画图划线
xx0 = np.linspace(-5, 5)  # (-5,5)之间x的值
yy0 = a0 * xx0 - (clf0.intercept_[0]) / w0[1]  # xx带入y，截距

# 画出与点相切的线
b0 = clf0.support_vectors_[0]
yy_down0 = a0 * xx0 + (b0[1] - a0 * b0[0])
b0 = clf0.support_vectors_[-1]
yy_up0 = a0 * xx0 + (b0[1] - a0 * b0[0])

print("W:", w0)
print("a:", a0)

print("\nsupport_vectors_:\n", clf0.support_vectors_)
print("clf.coef_:", clf0.coef_)

# 测试, 两个中括号！
for i in range(20):
    a = np.random.randn(1, 2) * 10
    print(a)
    print('测试', clf0.predict(a))

plt.figure(figsize=(8, 4))
plt.title("gamma:default")
plt.plot(xx0, yy0)
plt.plot(xx0, yy_down0)
plt.plot(xx0, yy_up0)
plt.scatter(clf0.support_vectors_[:, 0], clf0.support_vectors_[:, 1], s=80)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)  # [:，0]列切片，第0列

plt.axis('tight')

# plt.show()***********************************************************************

# svm设置：
clf1 = svm.SVC(C=2, kernel='linear', gamma=10)
clf1.fit(X, y)

# 获取w
w = clf1.coef_[0]
print('w', w)
a = -w[0] / w[1]  # 斜率
# 画图划线
xx = np.linspace(-5, 5)  # (-5,5)之间x的值
yy = a * xx - (clf1.intercept_[0]) / w[1]  # xx带入y，截距

# 画出与点相切的线
b = clf1.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf1.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

print("W:", w)
print("a:", a)

print("\nsupport_vectors_:\n", clf1.support_vectors_)
print("clf.coef_:", clf1.coef_)

# 测试, 两个中括号！
for i in range(20):
    a = np.random.randn(1, 2) * 10
    print(a)
    print('测试', clf1.predict(a))

plt.figure(figsize=(8, 4))
plt.plot(xx, yy)
plt.title("gamma=10")
plt.plot(xx, yy_down)
plt.plot(xx, yy_up)
plt.scatter(clf1.support_vectors_[:, 0], clf1.support_vectors_[:, 1], s=80)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)  # [:，0]列切片，第0列

plt.axis('tight')

plt.show()