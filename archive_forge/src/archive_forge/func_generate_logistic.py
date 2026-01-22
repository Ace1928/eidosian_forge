import numpy as np
from scipy.stats.distributions import norm
def generate_logistic():
    nclust = 100
    beta = np.array([1, -2, 1], dtype=np.float64)
    r = 0.4
    rx = 0.5
    re = 0.3
    p = len(beta)
    OUT = open('gee_logistic_1.csv', 'w', encoding='utf-8')
    for i in range(nclust):
        n = np.random.randint(3, 6)
        x = np.random.normal(size=(n, p))
        x = rx * np.random.normal() + np.sqrt(1 - rx ** 2) * x
        x[:, 2] = r * x[:, 1] + np.sqrt(1 - r ** 2) * x[:, 2]
        pr = 1 / (1 + np.exp(-np.dot(x, beta)))
        z = re * np.random.normal() + np.sqrt(1 - re ** 2) * np.random.normal(size=n)
        u = norm.cdf(z)
        y = 1 * (u < pr)
        for j in range(n):
            OUT.write('%d, %d,' % (i, y[j]))
            OUT.write(','.join(['%.3f' % b for b in x[j, :]]) + '\n')
    OUT.close()