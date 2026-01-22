import numpy as np
from scipy.stats.distributions import norm
def generate_poisson():
    beta = np.zeros(5, dtype=np.float64)
    beta[2] = 0.5
    beta[4] = -0.5
    nclust = 100
    OUT = open('gee_poisson_1.csv', 'w', encoding='utf-8')
    for i in range(nclust):
        n = np.random.randint(3, 6)
        x = np.random.normal(size=(n, 5))
        for j in range(5):
            x[:, j] += np.random.normal()
        lp = np.dot(x, beta)
        E = np.exp(lp)
        y = [np.random.poisson(e) for e in E]
        y = np.array(y)
        for j in range(n):
            OUT.write('%d, %d,' % (i, y[j]))
            OUT.write(','.join(['%.3f' % b for b in x[j, :]]) + '\n')
    OUT.close()