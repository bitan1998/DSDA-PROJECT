import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


names=['AGE','TB','DB','TP','Albumin','A/G','sgpt','sgot','ALKPHOS','GENDER']
dataset=pd.read_csv("Indian Liver Patient Dataset.csv")
##||REMOVING NAN FILES AS COLLEGE GAVE BAD DATASET||##
dataset1=dataset.dropna(subset = ['AGE','TB','DB','TP','Albumin','A/G','sgpt','sgot','ALKPHOS','GENDER'])


X=dataset1.iloc[:,:-1].values # REJECTING THE LAST COLUMN
y=dataset1.iloc[:,8].values
y=y.astype('int')## REMOVING CONTIGUOS FILES
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler ## BEST FOR  CLASSIFICATION TYPE MODEL
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

##BUILDING THE MODEL
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix  
print('||CONFUSION_MATRIX||')
print(confusion_matrix(y_test, y_pred))
print('\n') 
print('||CLASSIFICATION_REPORT||') 
print(classification_report(y_test, y_pred))

error = []

# Calculating error for K values between 1 and 100
for i in range(1, 100):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

#PLOT
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))  
plt.plot(range(1, 100), error, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 
plt.show()
