from math import sqrt
from csv import reader
from random import shuffle
from random import seed
from random import randrange

def mean(values):
	return sum(values)/len(values)

def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

def covariance(x, mean_x, y, mean_y):
	co_var = 0.0
	for i in range(len(x)):
		co_var += (x[i] - mean_x) * (y[i] - mean_y)
	return co_var

def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	mean_x, mean_y = mean(x), mean(y)
	variance_x = variance(x, mean_x)
	b1 = covariance(x, mean_x, y, mean_y) / variance_x
	b0 = mean_y - b1 * mean_x
	return b0, b1

def predict(b0, b1, x):
	return b0 + b1 * x

def simple_linear_regression(train, test):
	b0, b1 = coefficients(train)
	predictions = []
	for row in test:
		y = predict(b0, b1, row[0])
		predictions.append(y)
	return predictions

def rmse_metric(actual, predicted):
	error_sum = sum([(x - y) ** 2 for x, y in zip(actual, predicted)])
	error_mean = error_sum / len(actual)
	return sqrt(error_mean)

def evaluate_algorithm(dataset, algorithm, split):
	train_set, test_set = split_dataset(dataset, split)
	predicted = algorithm(train_set, test_set)
	actual = [row[1] for row in test_set]
	rmse = rmse_metric(actual, predicted)
	return rmse

def csv_to_dataset(filename):
	dataset = []
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			dataset.append(list(map(float, row)))
	return dataset

def split_dataset(dataset, split):
	#verify
	shuffle(dataset)
	train_size = int(split * len(dataset))
	train_set = dataset[:train_size]
	test_set = dataset[train_size:]
	return train_set, test_set

seed(1)
dataset = csv_to_dataset('insurance.csv')
split = 0.6
print(evaluate_algorithm(dataset, simple_linear_regression, split))