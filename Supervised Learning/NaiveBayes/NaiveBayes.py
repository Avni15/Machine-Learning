#importing libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

#loading breast cancer data
data=load_breast_cancer()
x=data.feature_names
y=data.target_names

#splitting data into test and training
x_train, x_test, y_train, y_test =train_test_split(data.data,data.target,test_size=0.3,random_state=1)

#making model object
model=GaussianNB()

#fitting model
model.fit(x_train,y_train)

#making predictions
pred=model.predict(x_test)

#calculating accuracy score
print(accuracy_score(y_test,pred)*100)