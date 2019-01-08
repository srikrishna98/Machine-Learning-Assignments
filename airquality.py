import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# importing the dataSet
url = 'datasets/AirQualityUCI.csv'
dataSet = pandas.read_csv(url,sep=';')
dataSet = dataSet.iloc[:10, :]

# splitting data as dependent and independent variables
x = dataSet.iloc[:, [2, 12, 13]].values
y = dataSet.iloc[:, 14].values

for i in range(x.shape[0]):
    x[i] = [float(m.replace(',', '.')) for m in x[i]]
for i in range(y.shape[0]):
    y[i] = float(y[i].replace(',', '.'))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# LogisticRegression
classifier = LinearRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(y_pred)