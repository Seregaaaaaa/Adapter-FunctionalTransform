import numpy as np
n = 1000
Size = 1000
M = np.zeros((n,Size+1))
for i in range(n):
    lambda_rate = np.random.uniform(0, 15000)
    M[i][0] = lambda_rate
    M[i][1:] = np.random.exponential(scale = 1/(lambda_rate), size=Size)
np.savetxt('/Users/sergei.samoilov/Documents/учеба/Учеба_3.2/ООАП/Adapter/data/PoisonData.csv', M, delimiter=',', comments='')
print('PoisonData.csv is save')