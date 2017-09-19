import pandas as pd
import numpy
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from random import randint

text_file = open("Output.csv", "w")
text_file.write("x1,x2,result\n")

for x in range(0, 10000):
    a = randint(1,100)
    b = randint(1,100)
    c = a + b
    text_file.write("%d, %d, %d\n" % (a, b, c))

text_file.close()






df = pd.read_csv('Output.csv', sep=',', header=0)

train, test = model_selection.train_test_split(df,test_size=0.01, random_state=0)

#clf = GaussianNB()
clf = SVC()
train_features = train.ix[:,0:2]
train_label = train.iloc[:,2]

clf.fit(train_features, train_label)

test_features = test.ix[:,0:2]
test_label = test.iloc[:,2]

test_data = pd.concat([test_features, test_label], axis=1)
test_data["prediction"] = clf.predict(test_features)


print(test_data)

print ("Naive Bayes Accuracy:", clf.score(test_features,test_label))


