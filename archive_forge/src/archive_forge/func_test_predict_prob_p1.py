from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
def test_predict_prob_p1(self):
    expected_params = [1, -0.5]
    np.random.seed(1234)
    nobs = 200
    exog = np.ones((nobs, 2))
    exog[:nobs // 2, 1] = 2
    mu_true = np.exp(exog.dot(expected_params))
    alpha = 0.05
    size = 1.0 / alpha * mu_true
    prob = size / (size + mu_true)
    endog = nbinom.rvs(size, prob, size=len(mu_true))
    res = sm.NegativeBinomialP(endog, exog).fit(disp=0)
    mu = res.predict()
    size = 1.0 / alpha * mu
    prob = size / (size + mu)
    probs = res.predict(which='prob')
    assert_allclose(probs, nbinom.pmf(np.arange(8)[:, None], size, prob).T, atol=0.01, rtol=0.01)
    probs_ex = res.predict(exog=exog[[0, -1]], which='prob')
    assert_allclose(probs_ex, probs[[0, -1]], rtol=1e-10, atol=1e-15)