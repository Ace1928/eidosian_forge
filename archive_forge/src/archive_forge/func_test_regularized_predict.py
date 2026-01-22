from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_regularized_predict():
    n = 100
    p = 5
    np.random.seed(3132)
    xmat = np.random.normal(size=(n, p))
    yvec = xmat.sum(1) + np.random.normal(size=n)
    wgt = np.random.uniform(1, 2, n)
    model_wls = WLS(yvec, xmat, weights=wgt)
    model_gls1 = GLS(yvec, xmat, sigma=np.diag(1 / wgt))
    model_gls2 = GLS(yvec, xmat, sigma=1 / wgt)
    res = []
    for model1 in [model_wls, model_gls1, model_gls2]:
        result1 = model1.fit_regularized(alpha=20.0, L1_wt=0.5, refit=True)
        res.append(result1)
        params = result1.params
        fittedvalues = np.dot(xmat, params)
        pr = model1.predict(result1.params)
        assert_allclose(fittedvalues, pr)
        assert_allclose(result1.fittedvalues, pr)
        pr = result1.predict()
        assert_allclose(fittedvalues, pr)
    assert_allclose(res[0].model.wendog, res[1].model.wendog, rtol=1e-10)
    assert_allclose(res[0].model.wexog, res[1].model.wexog, rtol=1e-10)
    assert_allclose(res[0].fittedvalues, res[1].fittedvalues, rtol=1e-10)
    assert_allclose(res[0].params, res[1].params, rtol=1e-10)
    assert_allclose(res[0].model.wendog, res[2].model.wendog, rtol=1e-10)
    assert_allclose(res[0].model.wexog, res[2].model.wexog, rtol=1e-10)
    assert_allclose(res[0].fittedvalues, res[2].fittedvalues, rtol=1e-10)
    assert_allclose(res[0].params, res[2].params, rtol=1e-10)