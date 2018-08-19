import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("train.csv").values

clf = DecisionTreeClassifier()

X_train = data[0:21000, 1:]
y_train = data[0:21000, 0]

clf.fit(X_train, y_train)
X_test = data[2100: , 1:]
y_test = data[2100: , 0]

sample = X_test[4]
sample.shape = (28, 28)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))
print("\n")
print(clf.predict([X_test[4]]))

plt.imshow(255 - sample, cmap = "gray")
plt.show()


