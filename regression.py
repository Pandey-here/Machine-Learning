import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()

# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module']
# print(diabetes.DESCR)

# diabetes_X=diabetes.data[:, np.newaxis,2] for linear regression means from all 
# features we have only taken 2 indexed feature

diabetes_X=diabetes.data    #for multiple regression
# print(diabetes_X)

diabetes_X_train=diabetes_X[:-30]   #taking 30 data from last
diabetes_X_test=diabetes_X[-30:]    #taking 30 data from begin

diabetes_Y_train=diabetes.target[:-30]  #label for last 30 data
diabetes_Y_test=diabetes.target[-30:]   #label for first 30 data
#here target means label thus important to match the labels with their data

model=linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_Y_train)    #training the model

diabetes_Y_predict= model.predict(diabetes_X_test)  #predicting value

print("Mean square error is: ",mean_squared_error(diabetes_Y_test,diabetes_Y_predict))

print("Weights: ",model.coef_)
print("Intercepts: ",model.intercept_)

# we can only plot graph for linear regression
# plt.scatter(diabetes_X_test,diabetes_Y_test)    #graph:- dataset vs label
# plt.plot(diabetes_X_test,diabetes_Y_predict)
# plt.show()


# using linear regression
# Mean square error is:  3035.060115291269
# Weights:  [941.43097333]
# Intercepts:  153.39713623331644

# using multiple regression
# Mean square error is:  1826.4841712795044
# Weights:  [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067
#   458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ]
# Intercepts:  153.05824267739402

# diabetes_X_train-> datas for train model
# diabetes_Y_train-> labels w.r.t to diabetes_X_train
# diabetes_X_test-> datas for test the model 
# diabetes_Y_test-> labels w.r.t to diabetes_Y_test
# diabetes_Y_predict-> result that model has predicted if result close to diabetes_Y_test
# then model is great 
# weights->w1,w2,w3... coeficient prefer video calulated on basis of diabetes_X_train
# and diabetes_Y_train
# intercept-> w0


# model which have less mean_squared_error that model is most accurate
# multiple regression provides more close results than linear regression

# linear regression:- only one feature
# multiple regression:- multiple feature

# graph can be plotted in linear regression bcoz only one variable or feature
# not possible in multiple bcoz result will lead to plane not 2d graph