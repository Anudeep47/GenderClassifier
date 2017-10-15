from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import  neighbors
from sklearn.metrics import accuracy_score

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

print("accuracy_score for DecisionTreeClassifier : %f" %(accuracy_score(Y, prediction)))
print("accuracy_score for svm : %f" %(accuracy_score(Y, prediction1)))
print("accuracy_score for GaussianNB : %f" %(accuracy_score(Y, prediction2)))
print("accuracy_score for KNeighborsClassifier : %f" %(accuracy_score(Y, prediction3)))
