import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score



f = open("fakeNews.csv", "r")

temp_data = f.readlines()
f.close()

x = []

for i in range(0, len(temp_data)):

    p = temp_data[i].split(";")
    x.append(p)

data = []
labels = []    
    
for i in range(0, len(x)):
    if(len(x[i])<2):
        continue
    if x[i][0] not in data:
        data.append(x[i][0])
        labels.append(x[i][1])
        

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33)

tfidf_vectorizer = TfidfVectorizer() 
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)



RandFor = RandomForestClassifier()
RandFor.fit(tfidf_train, y_train)
y_pred = RandFor.predict(tfidf_test)

print("Random Forset : ", accuracy_score(y_test, y_pred)*100.0)

