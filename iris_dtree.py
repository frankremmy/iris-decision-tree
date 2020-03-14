import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics


col_names = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

Iris = pd.read_csv('Iris.csv', sep=',')
print(Iris.head())


features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = Iris[features]
y = Iris.Species

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% to train and 30% to test

print('The length of your Training data set (70%) is:' + str(len(X_train)) + '\n')

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Predicted species are most likely ' + str(y_pred) + '\n')
# Testing model Accuracy
print("Model Accuracy:", metrics.accuracy_score(y_test, y_pred))