import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
def gen_tweedie(p):
    np.random.seed(3242)
    n = 500
    x = np.random.normal(size=(n, 4))
    lpr = np.dot(x, np.r_[1, -1, 0, 0.5])
    mu = np.exp(lpr)
    lam = 10 * mu ** (2 - p) / (2 - p)
    alp = (2 - p) / (p - 1)
    bet = 10 * mu ** (1 - p) / (p - 1)
    y = np.empty(n)
    N = np.random.poisson(lam)
    for i in range(n):
        y[i] = np.random.gamma(alp, 1 / bet[i], N[i]).sum()
    return (y, x)