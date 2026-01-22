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
def test_null_options():
    nobs = 10
    exog = np.ones((20, 2))
    exog[:nobs // 2, 1] = 0
    mu = np.exp(exog.sum(1))
    endog = np.random.poisson(mu)
    res = Poisson(endog, exog).fit(start_params=np.log([1, 1]), disp=0)
    llnull0 = res.llnull
    assert_(hasattr(res, 'res_llnull') is False)
    res.set_null_options(attach_results=True)
    lln = res.llnull
    assert_allclose(res.res_null.mle_settings['start_params'], np.log(endog.mean()), rtol=1e-10)
    assert_equal(res.res_null.mle_settings['optimizer'], 'bfgs')
    assert_allclose(lln, llnull0)
    res.set_null_options(attach_results=True, start_params=[0.5], method='nm')
    lln = res.llnull
    assert_allclose(res.res_null.mle_settings['start_params'], [0.5], rtol=1e-10)
    assert_equal(res.res_null.mle_settings['optimizer'], 'nm')
    res.summary()
    assert_('prsquared' in res._cache)
    assert_equal(res._cache['llnull'], lln)
    assert_('prsquared' in res._cache)
    assert_equal(res._cache['llnull'], lln)
    res.set_null_options(llnull=999)
    assert_('prsquared' not in res._cache)
    assert_equal(res._cache['llnull'], 999)