from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading data sets
iris=datasets.load_iris()

#printing description and feature
# print(iris.DESCR)

features=iris.data
label=iris.target

#training classifier
clf=KNeighborsClassifier()
clf.fit(features,label)

preds=clf.predict([[31,1,1,1]])
print(preds)