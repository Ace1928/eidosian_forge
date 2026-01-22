import numpy as np
from scipy.stats.distributions import norm
def generate_nested_linear():
    nclust = 200
    beta = np.array([1, -2, 1], dtype=np.float64)
    v1 = 1
    v2 = 0.5
    v3 = 1.5
    p = len(beta)
    OUT = open('gee_nested_linear_1.csv', 'w', encoding='utf-8')
    for i in range(nclust):
        x = np.random.normal(size=(10, p))
        y = np.dot(x, beta)
        y += np.sqrt(v1) * np.random.normal()
        y[0:5] += np.sqrt(v2) * np.random.normal()
        y[5:10] += np.sqrt(v2) * np.random.normal()
        y += np.sqrt(v3) * np.random.normal(size=10)
        for j in range(10):
            OUT.write('%d, %.3f,' % (i, y[j]))
            OUT.write(','.join(['%.3f' % b for b in x[j, :]]) + '\n')
    OUT.close()