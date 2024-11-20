# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:32:10 2024

@author: TANISHQ
"""

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# Importing Dataset
dataset = pd.read_csv(r"E:\NIT DataSci Notes\36.LOGISTIC REGRESSION CODE\logit classification.csv")
#Here in this car dealer dataset we have userid,gender, age, salary, purchased (yes/no)

X = dataset.iloc[:,[2,3]].values #Independent Values

y = dataset.iloc[:,4].values  #Dependent Values


#Split data in training and testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


#Scale data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)


training_score = classifier.score(X_train,y_train)
print("Bias: ", training_score)
testing_score = classifier.score(X_test,y_test)
print("Variance: ", testing_score)


# FUTURE PREDICTION 

dataset1 = pd.read_csv(r"E:\NIT DataSci Notes\36.LOGISTIC REGRESSION CODE\final1.csv")
#Here in this car dealer dataset we have userid,gender, age, salary.
#here we have to predict, will they purchase Yes or No

d2 = dataset1.copy()
#This creates a copy of dataset1 and assigns it to d2. This way, you keep the original dataset intact while modifying d2.

dataset1 = dataset1.iloc[:,[3,4]].values  # Here we tool X values only

from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

M = sc.fit_transform(dataset1)

d2 ['y_pred'] = classifier.predict(M)
# Here first we put out X that is scaled in M to classifer and then put those values in 
# d2 file in column name y_pred
d2.to_csv('pred_model.csv') # We saved d2 as csv with given name

import os
print("Current Working Directory:", os.getcwd())











