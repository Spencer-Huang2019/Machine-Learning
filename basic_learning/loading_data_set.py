
from sklearn.datasets import load_boston,load_iris,load_breast_cancer,load_diabetes,load_digits,load_wine

boston = load_boston()
print("boston data shape: ", boston.data.shape)
iris = load_iris()
print("iris data shape: ", iris.data.shape)
cancer = load_breast_cancer()
print("cancer data shape: ", cancer.data.shape)
diabetes = load_diabetes()
print("diabetes data shape: ", diabetes.data.shape)
digits = load_digits()
print("digits data shape: ", digits.data.shape)
wine = load_wine()
print("wine data shape: ", wine.data.shape)


#clear the basic_learning under sklearn ENV
# datasets.clear_data_home()

# for linear regression:Boston,california_housing,diabetes
#import the boston dataset from sklearn
# dataset = datasets.load_boston()
#data,target = datasets.load_boston(return_X_y=True)

# #check the shape of dataset
# print(dataset.data.shape)
# print(dataset.target.shape)
#
# #check the features of dataset
# print(dataset.feature_names)
# print(dataset.DESCR)

# #get dataset
# data = dataset.data[:, 5]
# print(data)
# print(data.reshape(-1,1))

