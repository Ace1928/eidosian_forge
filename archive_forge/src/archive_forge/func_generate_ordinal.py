import numpy as np
from scipy.stats.distributions import norm
def generate_ordinal():
    beta = np.zeros(5, dtype=np.float64)
    beta[2] = 1
    beta[4] = -1
    rz = 0.5
    OUT = open('gee_ordinal_1.csv', 'w', encoding='utf-8')
    for i in range(200):
        n = np.random.randint(3, 6)
        x = np.random.normal(size=(n, 5))
        for j in range(5):
            x[:, j] += np.random.normal()
        pr = np.dot(x, beta)
        pr = np.array([1, 0, -0.5]) + pr[:, None]
        pr = 1 / (1 + np.exp(-pr))
        z = rz * np.random.normal() + np.sqrt(1 - rz ** 2) * np.random.normal(size=n)
        u = norm.cdf(z)
        y = (u[:, None] > pr).sum(1)
        for j in range(n):
            OUT.write('%d, %d,' % (i, y[j]))
            OUT.write(','.join(['%.3f' % b for b in x[j, :]]) + '\n')
    OUT.close()