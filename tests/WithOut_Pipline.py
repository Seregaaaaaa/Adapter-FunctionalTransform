import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.linear_model import LinearRegression
from ..helpers.utils import print_result

data = np.genfromtxt('PoisonData.csv', delimiter=',', skip_header=0)

n, Size = data.shape

train = data[:int(n*0.8), :]
test = data[int(n*0.8):, :]
test = test[test[:, 0].argsort()]
train_X = train[:, 1:]
train_Y = train[:, 0]
test_X = test[:, 1:]
test_Y = test[:, 0]

log_train_X = np.log(train_X)
log_test_X = np.log(test_X)

log_linear_model = LinearRegression()
log_linear_model.fit(log_train_X, train_Y)

predictions = log_linear_model.predict(log_test_X)

print_result(predictions, test_Y)
