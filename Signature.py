import numpy as np
import  urllib.request

from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn import metrics
from sklearn.metrics import accuracy_score





f1 = open("dutch_train.csv", "r")
train_data = f1.readlines()
f1.close()
print(train_data)

dataset1 = np.loadtxt(train_data, delimiter = ',')

x_train = dataset1[:, 0:19999]
y_train = dataset1[:, 19999]



f2 = open("dutch_test.csv", "r")
test_data = f2.readlines()
f2.close()


dataset2 = np.loadtxt(test_data, delimiter = ',')


x_test = dataset2[:, 0:19999]
y_test = dataset2[:, 19999]



SVM = svm.SVC(gamma=10)
SVM.fit(x_train, y_train)

y_pred = SVM.predict(x_test)

print("SVM : ", accuracy_score(y_test, y_pred)*100.0)


Multi = MultinomialNB()
Multi.fit(x_train, y_train)
y_pred = Multi.predict(x_test)

print("Naive Bayes : ", accuracy_score(y_test, y_pred)*100.0)


RandFor = RandomForestClassifier()
RandFor.fit(x_train, y_train)
y_pred = RandFor.predict(x_test)

print("Random Forset : ", accuracy_score(y_test, y_pred)*100.0)
