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
def test_ivgmm0_r():
    n, k = exog.shape
    nobs, k_instr = instrument.shape
    w0inv = np.dot(instrument.T, instrument) / nobs
    w0 = np.linalg.inv(w0inv)
    mod = gmm.IVGMM(endog, exog, instrument)
    res = mod.fit(np.ones(exog.shape[1], float), maxiter=0, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol': 1e-08, 'disp': 0})
    assert_allclose(res.params, params, rtol=0.0001, atol=0.0001)
    assert_allclose(res.bse, bse, rtol=0.09, atol=0)
    score = res.model.score(res.params, w0)
    assert_allclose(score, np.zeros(score.shape), rtol=0, atol=5e-06)