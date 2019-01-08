import matplotlib.pyplot as plt
import pandas


def mean(val):
    return sum(val)/float(len(val))


def variance(val, lmean):
    return sum([(x-lmean)**2 for x in val])


def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]


def simple_linear_regression(train_d, test_d):
    predictions = list()
    b0, b1 = coefficients(train_d)
    for row in test_d:
        yp = b0 + b1 * row[0]
        predictions.append(yp)
    return predictions


url = 'datasets/zomato.csv'
dataSet = pandas.read_csv(url)
cost = dataSet.iloc[:500, 10:11].values
rating = dataSet.iloc[:500, 17:18].values
train = [[cost[i], rating[i]] for i in range(cost.size-325)]
test = [[cost[i], rating[i]] for i in range(cost.size-325, cost.size)]

print("Test data:")
print(test)
# print("Predictions:")
# print(simple_linear_regression(train,test))

plt.scatter(cost[:175], rating[:175], color='blue')
# plt.plot(sep_len[:15],pet_len[:15],color = 'red')
plt.plot(cost[:175], simple_linear_regression(train, train), color='green')
plt.title('Cost vs Rating (Training Set)')
plt.xlabel('Average Cost for two')
plt.ylabel('Rating')
# plt.axis([4, 8, 0, 6])
plt.show()

plt.scatter(cost[175:], rating[175:], color='blue')
# plt.plot(sep_len[15:],pet_len[15:],color = 'red')
plt.plot(cost[175:], simple_linear_regression(train, test), color='green')
plt.title('Cost vs Rating (Testing Set)')
plt.xlabel('Average Cost for two')
plt.ylabel('Rating')
# plt.axis([4, 8, 0, 6])
plt.show()
