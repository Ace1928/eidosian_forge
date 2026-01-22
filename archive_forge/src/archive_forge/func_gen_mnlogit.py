import numpy as np
from statsmodels.discrete.conditional_models import (
from statsmodels.tools.numdiff import approx_fprime
from numpy.testing import assert_allclose
import pandas as pd
def gen_mnlogit(n):
    np.random.seed(235)
    g = np.kron(np.ones(5), np.arange(n // 5))
    x1 = np.random.normal(size=n)
    x2 = np.random.normal(size=n)
    xm = np.concatenate((x1[:, None], x2[:, None]), axis=1)
    pa = np.array([[0, 1, -1], [0, 2, -1]])
    lpr = np.dot(xm, pa)
    pr = np.exp(lpr)
    pr /= pr.sum(1)[:, None]
    cpr = pr.cumsum(1)
    y = 2 * np.ones(n)
    u = np.random.uniform(size=n)
    y[u < cpr[:, 2]] = 2
    y[u < cpr[:, 1]] = 1
    y[u < cpr[:, 0]] = 0
    df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'g': g})
    return df