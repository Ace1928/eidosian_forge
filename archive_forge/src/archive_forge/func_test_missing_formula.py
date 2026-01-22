from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
def test_missing_formula(self):
    np.random.seed(34234)
    endog = np.random.normal(size=100)
    exog1 = np.random.normal(size=100)
    exog2 = np.random.normal(size=100)
    exog3 = np.random.normal(size=100)
    groups = np.kron(lrange(20), np.ones(5))
    endog[0] = np.nan
    endog[5:7] = np.nan
    exog2[10:12] = np.nan
    data0 = pd.DataFrame({'endog': endog, 'exog1': exog1, 'exog2': exog2, 'exog3': exog3, 'groups': groups})
    for k in (0, 1):
        data = data0.copy()
        kwargs = {}
        if k == 1:
            data['offset'] = 0
            data['time'] = 0
            kwargs['offset'] = 'offset'
            kwargs['time'] = 'time'
        mod1 = gee.GEE.from_formula('endog ~ exog1 + exog2 + exog3', groups='groups', data=data, missing='drop', **kwargs)
        rslt1 = mod1.fit()
        assert_almost_equal(len(mod1.endog), 95)
        assert_almost_equal(np.asarray(mod1.exog.shape), np.r_[95, 4])
        data = data.dropna()
        kwargs = {}
        if k == 1:
            kwargs['offset'] = data['offset']
            kwargs['time'] = data['time']
        mod2 = gee.GEE.from_formula('endog ~ exog1 + exog2 + exog3', groups=data['groups'], data=data, missing='none', **kwargs)
        rslt2 = mod2.fit()
        assert_almost_equal(rslt1.params.values, rslt2.params.values)
        assert_almost_equal(rslt1.bse.values, rslt2.bse.values)