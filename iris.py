import pandas
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def generateclassificationreport(test, pred):
    print(classification_report(test, pred))
    print(confusion_matrix(test, pred))
    print('accuracy is ', accuracy_score(test, pred))


# importing the dataSet
url = 'iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataSet = pandas.read_csv(url, names=names)

# splitting data as dependent and independent variables
x = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

generateclassificationreport(y_test, y_pred)
