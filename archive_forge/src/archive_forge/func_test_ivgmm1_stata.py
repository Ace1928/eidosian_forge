from statsmodels.compat.python import lrange, lmap
import os
import copy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
import statsmodels.sandbox.regression.gmm as gmm
def test_ivgmm1_stata():
    params_stata = np.array([4.0335099, 0.17242531, -0.00909883, 0.04928949, 0.04221709, -0.10179345, 0.12611095, -0.05961711, 0.04867956, 0.15281763, 0.17443605, 0.09166597, 0.09323976])
    bse_stata = np.array([0.33503289, 0.02073947, 0.00488624, 0.0080498, 0.00946363, 0.03371053, 0.03081138, 0.05171372, 0.04981322, 0.0479285, 0.06112515, 0.0554618, 0.06084901])
    n, k = exog.shape
    nobs, k_instr = instrument.shape
    w0inv = np.dot(instrument.T, instrument) / nobs
    w0 = np.linalg.inv(w0inv)
    start = OLS(endog, exog).fit().params
    mod = gmm.IVGMM(endog, exog, instrument)
    res = mod.fit(start, maxiter=1, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol': 1e-06, 'disp': 0})