#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.metrics import accuracy_score
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clfrf = RandomForestClassifier(n_estimators=10)

t0 = time()
clfrf = clfrf.fit(features_train, labels_train)
print "RandomForestClassifier training time: ", round(time() - t0, 3), "s"

t0 = time()
pred = clfrf.predict(features_test)
print "RandomForestClassifier prediction time: ", round(time() - t0, 3), "s"

print "RandomForestClassifier(10): ", accuracy_score(pred, labels_test)

# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
clfad = AdaBoostClassifier(n_estimators=50)

t0 = time()
clfad = clfad.fit(features_train, labels_train)
print "AdaBoostClassifier training time: ", round(time() - t0, 3), "s"

t0 = time()
pred = clfad.predict(features_test)
print "AdaBoostClassifier prediction time: ", round(time() - t0, 3), "s"

print "AdaBoostClassifier(50): ", accuracy_score(pred, labels_test)

# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
clfk = KNeighborsClassifier(n_neighbors=4)

t0 = time()
clfk = clfk.fit(features_train, labels_train)
print "KNeighborsClassifier training time: ", round(time() - t0, 3), "s"

t0 = time()
pred = clfk.predict(features_test)
print "KNeighborsClassifier prediction time: ", round(time() - t0, 3), "s"

print "KNeighborsClassifier(4): ", accuracy_score(pred, labels_test)






try:
    #prettyPicture(clfk, features_test, labels_test)
    #prettyPicture(clfad, features_test, labels_test)
    prettyPicture(clfrf, features_test, labels_test)
except NameError:
    pass
