#train a logistic regression classifier to predict whether a flower is iris virginica or not
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


iris=datasets.load_iris()
# print(iris.DESCR)

iris_data=iris.data[:,np.newaxis,3]
iris_target=(iris.target==2).astype(np.int)

# print(iris_target)
# print(iris_data)

#training a logistic regression classifier
clf=LogisticRegression()
clf.fit(iris_data,iris_target)
example=clf.predict([[2.6]])
print(example)
