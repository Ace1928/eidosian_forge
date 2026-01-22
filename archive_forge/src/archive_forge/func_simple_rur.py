from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import MONTH_END, YEAR_END, assert_index_equal
from statsmodels.compat.platform import PLATFORM_WIN
from statsmodels.compat.python import lrange
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas import DataFrame, Series, date_range
import pytest
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.datasets import macrodata, modechoice, nile, randhie, sunspots
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import array_like, bool_like
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
def simple_rur(self, x, store=False):
    x = array_like(x, 'x')
    store = bool_like(store, 'store')
    nobs = x.shape[0]
    if nobs != x.size:
        raise ValueError(f'x of shape {x.shape} not understood')
    pvals = [0.01, 0.025, 0.05, 0.1, 0.9, 0.95]
    n = np.array([25, 50, 100, 150, 200, 250, 500, 1000, 2000, 3000, 4000, 5000])
    crit = np.array([[0.6626, 0.8126, 0.9192, 1.0712, 2.4863, 2.7312], [0.7977, 0.9274, 1.0478, 1.1964, 2.6821, 2.9613], [0.907, 1.0243, 1.1412, 1.2888, 2.8317, 3.1393], [0.9543, 1.0768, 1.1869, 1.3294, 2.8915, 3.2049], [0.9833, 1.0984, 1.2101, 1.3494, 2.9308, 3.2482], [0.9982, 1.1137, 1.2242, 1.3632, 2.9571, 3.2482], [1.0494, 1.1643, 1.2712, 1.4076, 3.0207, 3.3584], [1.0846, 1.1959, 1.2988, 1.4344, 3.0653, 3.4073], [1.1121, 1.22, 1.323, 1.4556, 3.0948, 3.4439], [1.1204, 1.2295, 1.3318, 1.4656, 3.1054, 3.4632], [1.1309, 1.2347, 1.3318, 1.4693, 3.1165, 3.4717], [1.1377, 1.2402, 1.3408, 1.4729, 3.1252, 3.4807]])
    inter_crit = np.zeros((1, crit.shape[1]))
    for i in range(crit.shape[1]):
        f = interp1d(n, crit[:, i])
        inter_crit[0, i] = f(nobs)
    count = 0
    max_p = x[0]
    min_p = x[0]
    for v in x[1:]:
        if v > max_p:
            max_p = v
            count = count + 1
        if v < min_p:
            min_p = v
            count = count + 1
    rur_stat = count / np.sqrt(len(x))
    k = len(pvals) - 1
    for i in range(len(pvals) - 1, -1, -1):
        if rur_stat < inter_crit[0, i]:
            k = i
        else:
            break
    p_value = pvals[k]
    warn_msg = '        The test statistic is outside of the range of p-values available in the\n        look-up table. The actual p-value is {direction} than the p-value returned.\n        '
    direction = ''
    if p_value == pvals[-1]:
        direction = 'smaller'
    elif p_value == pvals[0]:
        direction = 'larger'
    if direction:
        warnings.warn(warn_msg.format(direction=direction), InterpolationWarning)
    crit_dict = {'10%': inter_crit[0, 3], '5%': inter_crit[0, 2], '2.5%': inter_crit[0, 1], '1%': inter_crit[0, 0]}
    if store:
        from statsmodels.stats.diagnostic import ResultsStore
        rstore = ResultsStore()
        rstore.nobs = nobs
        rstore.H0 = 'The series is not stationary'
        rstore.HA = 'The series is stationary'
        return (rur_stat, p_value, crit_dict, rstore)
    else:
        return (rur_stat, p_value, crit_dict)