#!/usr/bin/python

# Author: Ajay Bhatia
# 
# Script to demonstrate State Vector Classifier State Vector Machine
# to recognize digit by training it based on sklearn data and library

# import pyplot
import matplotlib.pyplot as plt
# import example dataset from sklearn 
from sklearn import datasets
# import svm from sklearn
from sklearn import svm

# load digits sample data from datasets in sklearn
digits = datasets.load_digits()

# create SVC SVM classifier
clf = svm.SVC(gamma=0.0001, C=100)

# x = data to the last index, y = target to the last index
x, y = digits.data[:-1], digits.target[:-1]

# train SVM
clf.fit(x, y)

# predict 300th from the last
print('Prediction: ', clf.predict(digits.data[-300]))

# plot the last image 
plt.imshow(digits.images[-300], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
