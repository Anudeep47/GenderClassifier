from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import  neighbors
from sklearn.metrics import accuracy_score
import numpy as np

clf = tree.DecisionTreeClassifier()
clf1 = svm.SVC()
clf2 = GaussianNB()
clf3 = neighbors.KNeighborsClassifier()

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

prediction = clf.predict(X)
prediction1 = clf1.predict(X)
prediction2 = clf2.predict(X)
prediction3 = clf3.predict(X)

accuracy = (accuracy_score(Y, prediction))*100
accuracy1 = (accuracy_score(Y, prediction1))*100
accuracy2 = (accuracy_score(Y, prediction2))*100
accuracy3 = (accuracy_score(Y, prediction3))*100

print("accuracy_score for DecisionTreeClassifier : %f" %accuracy)
print("accuracy_score for svm : %f" %accuracy1)
print("accuracy_score for GaussianNB : %f" %accuracy2)
print("accuracy_score for KNeighborsClassifier : %f" %accuracy3)


classifiers = {0: "SVM", 1: "Naive Bayes", 2: "KNeighbors"}
index = np.argmax([accuracy1, accuracy2, accuracy3])
print("Best gender classifier is {}".format(classifiers[index]))
