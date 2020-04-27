#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
print("python:{}".format(sys.version))
import scipy
print("scipy:{}".format(scipy.version))
import numpy
import pandas
import matplotlib
import sklearn


# In[14]:


import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# In[26]:


# loading the data
names=['sepal-length','sepal-width','petal-length','petal-width','class']
data=pandas.read_csv('iris.csv')


# In[20]:


data.head()


# In[21]:


# statistical summary
print(data.describe())


# In[28]:


# dimension of the data
print(data.shape)


# In[29]:


print(data.head(20))


# In[30]:


# class distribution
print(data.groupby('variety').size())


# In[31]:


# visualize the data
data.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
pyplot.show()


# In[32]:


#histogram of this variable
data.hist()
pyplot.show()


# In[33]:


# multivarient plot
scatter_matrix(data)
pyplot.show()


# In[34]:


# creating a validation data
# splitting data
array=data.values
x=array[:,0:4]
y=array[:, 4]
x_train, x_validation, y_train, y_validation=train_test_split(x,y,test_size=0.2,random_state=1)


# In[42]:


#Logistic Regression
#Linear Discriminant Analysis
#K-Nearest neighbors
#Classification and Regression Trees
#Gaussian Naive Bayes
# support Vector Machine


# building models
models=[]
models.append(("LR",LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))                                       


# In[ ]:




