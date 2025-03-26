import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

def print_result(predictions, test_Y, name:str = ''):
    mape = np.mean(np.abs((test_Y - predictions) / test_Y)) * 100
    r2 = r2_score(test_Y, predictions)
    mse = mean_squared_error(test_Y, predictions)
    rmse = np.sqrt(mean_squared_error(test_Y, predictions))

    print(name, " RESULT")
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R² Score: {r2}')
    print(f'Mean Absolute Percentage Error: {mape}%')