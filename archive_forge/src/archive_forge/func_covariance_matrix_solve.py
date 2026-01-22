from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
@Appender(CovStruct.covariance_matrix_solve.__doc__)
def covariance_matrix_solve(self, expval, index, stdev, rhs):
    k = len(expval)
    r = self.dep_params
    soln = []
    if k == 1:
        return [x / stdev ** 2 for x in rhs]
    if k == 2:
        mat = np.array([[1, -r], [-r, 1]])
        mat /= 1.0 - r ** 2
        for x in rhs:
            if x.ndim == 1:
                x1 = x / stdev
            else:
                x1 = x / stdev[:, None]
            x1 = np.dot(mat, x1)
            if x.ndim == 1:
                x1 /= stdev
            else:
                x1 /= stdev[:, None]
            soln.append(x1)
        return soln
    c0 = (1.0 + r ** 2) / (1.0 - r ** 2)
    c1 = 1.0 / (1.0 - r ** 2)
    c2 = -r / (1.0 - r ** 2)
    soln = []
    for x in rhs:
        flatten = False
        if x.ndim == 1:
            x = x[:, None]
            flatten = True
        x1 = x / stdev[:, None]
        z0 = np.zeros((1, x1.shape[1]))
        rhs1 = np.concatenate((x1[1:, :], z0), axis=0)
        rhs2 = np.concatenate((z0, x1[0:-1, :]), axis=0)
        y = c0 * x1 + c2 * rhs1 + c2 * rhs2
        y[0, :] = c1 * x1[0, :] + c2 * x1[1, :]
        y[-1, :] = c1 * x1[-1, :] + c2 * x1[-2, :]
        y /= stdev[:, None]
        if flatten:
            y = np.squeeze(y)
        soln.append(y)
    return soln