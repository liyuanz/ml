#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np


## simulate date
np.random.seed(3) # fix data seed
df= pd.DataFrame({'labels':np.random.randint(0,2,200), 
                  'x1':np.random.uniform(-10,10,200), 
                  'x2':np.random.uniform(5,10,200) })
x=np.array(df.iloc[:,[1,2]])
y=np.array(df.iloc[:,0])  


# define functions 
def sigmoid(x):
    return 1/(1+np.exp(-x))  # replaced math.exp by numpy.exp
def loss(x, y, theta):
    return (-np.dot(y,np.log(sigmoid(np.dot(x,theta))))-np.dot(1-y, np.log(1-sigmoid(np.dot(x,theta)))))/200
def gradient(x, y, theta):
    return np.dot(x.T,(sigmoid(np.dot(x,theta)) - y))/200
def logit(x, y, theta):
    alpha=0.01
    while True:
        theta_old=theta
        theta=theta-alpha*gradient(x, y, theta)
        if abs(loss(x,y,theta)-loss(x,y,theta_old))<1e-10:  # used a smaller stopping criterion
            return theta
            break
def predict_prob(theta):
    return sigmoid(np.dot(x,theta))

## add predict, accuracy and main functions below
def predict(theta, threshold):
    return (predict_prob(theta)>threshold).astype(int)
def accuracy(theta, threshold=0.5):
    return (y==predict(theta, threshold)).sum()/len(y)
def main():
    theta=np.array([0,0]) 
    theta_ = logit(x, y, theta)
    print('The estimate of theta is:', theta_)
    print('The accuracy is:', accuracy(theta_))

if __name__=='__main__':
    main()
#The estimate of theta is: [ 0.03481532 -0.01979547]
#The accuracy is: 0.56

# compare estimates with sklearn model

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(fit_intercept=False)
clf.fit(x, y)
print(clf.coef_)
#[[ 0.03480515 -0.01978841]]


