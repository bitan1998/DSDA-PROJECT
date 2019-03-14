import numpy as np
from sklearn.svm import SVR
import pandas

dataset= pandas.read_csv("Indian Liver Patient Dataset.csv")
##||REMOVING NAN FILES AS COLLEGE GAVE BAD DATASET||##
dataset1=dataset.dropna(subset = ['AGE','TB','DB','TP','Albumin','A/G','sgpt','sgot','ALKPHOS','GENDER'])

X=dataset1.iloc[:,:-1].values # REJECTING THE LAST COLUMN
y=dataset1.iloc[:,6].values
y=y.astype('int')## REMOVING CONTIGUOS FILES
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler ## BEST FOR  CLASSIFICATION TYPE MODEL
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
y_poly = svr_poly.fit(X_train, y_train).predict(X_test)

print("\n||MODEL SCORE||\n")
print('MODEL_RBF:',svr_rbf.score(X_train,y_train))
print('MODEL_LIN:',svr_lin.score(X_train,y_train))
print('MODEL_POLY:',svr_poly.score(X_train,y_train))

x=np.std(np.square(np.subtract(y_test,y_rbf)))
y=np.std(np.square(np.subtract(y_test,y_lin)))
z=np.std(np.square(np.subtract(y_test,y_poly)))

print("\n||MODEL RMSE||\n")
print('RMSE_RBF::',x)
print('RMSE_LIN::',y)
print('RMSE_POLY::',z)
