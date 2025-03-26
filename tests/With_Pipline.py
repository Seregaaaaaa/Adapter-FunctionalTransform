import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.linear_model import LinearRegression
from src.Pipeline import pipeline_Log_LR
from helpers.utils import print_result

data = np.genfromtxt('data/PoisonData.csv', delimiter=',', skip_header=0)

n, Size = data.shape

train = data[:int(n*0.8), :]
test = data[int(n*0.8):, :]
test = test[test[:, 0].argsort()]
train_X = train[:, 1:]
train_Y = train[:, 0]

test_X = test[:, 1:]
test_Y = test[:, 0]

pipeline_Log_LR.fit(train_X, train_Y)
predictions = pipeline_Log_LR.predict(test_X)

print_result(predictions, test_Y)

#PYTHONPATH=/Users/sergei.samoilov/Documents/учеба/Учеба_3.2/ООАП/Adapter python3 tests/With_Pipline.py;  
