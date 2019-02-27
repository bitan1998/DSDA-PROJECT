import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

names=['AGE','TB','DB','TP','Albumin','A/G','sgpt','sgot','ALKPHOS','GENDER']
dataset=pd.read_csv("Indian Liver Patient Dataset.csv")
##||REMOVING NAN FILES AS COLLEGE GAVE BAD DATASET||##
dataset1=dataset.dropna(subset = ['AGE','TB','DB','TP','Albumin','A/G','sgpt','sgot','ALKPHOS','GENDER'])

X=dataset1.iloc[:,:-1].values # REJECTING THE LAST COLUMN
y=dataset1.iloc[:,8].values
y=y.astype('int')## REMOVING CONTIGUOS FILES

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate the classifier
gnb =MultinomialNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print('Model Score:',gnb.score(X_train,y_train))

mean_survival=np.mean(X_train)
mean_not_survival=100-mean_survival
print("SUCCESS = {:03.2f}%, FAILURE = {:03.2f}%".format(mean_survival,mean_not_survival))

from sklearn.metrics import classification_report, confusion_matrix  
print('||CONFUSION_MATRIX||')
print(confusion_matrix(y_test, y_pred))
print('\n') 
print('||CLASSIFICATION_REPORT||') 
print(classification_report(y_test, y_pred))

