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
def test_autoregressive(self):
    dep_params_true = [0, 0.589208623896, 0.559823804948]
    params_true = [[1.08043787, 1.12709319, 0.90133927], [0.9613677, 1.05826987, 0.90832055], [1.05370439, 0.96084864, 0.93923374]]
    np.random.seed(342837482)
    num_group = 100
    ar_param = 0.5
    k = 3
    ga = families.Gaussian()
    for gsize in (1, 2, 3):
        ix = np.arange(gsize)[:, None] - np.arange(gsize)[None, :]
        ix = np.abs(ix)
        cmat = ar_param ** ix
        cmat_r = np.linalg.cholesky(cmat)
        endog = []
        exog = []
        groups = []
        for i in range(num_group):
            x = np.random.normal(size=(gsize, k))
            exog.append(x)
            expval = x.sum(1)
            errors = np.dot(cmat_r, np.random.normal(size=gsize))
            endog.append(expval + errors)
            groups.append(i * np.ones(gsize))
        endog = np.concatenate(endog)
        groups = np.concatenate(groups)
        exog = np.concatenate(exog, axis=0)
        ar = cov_struct.Autoregressive(grid=False)
        md = gee.GEE(endog, exog, groups, family=ga, cov_struct=ar)
        mdf = md.fit()
        assert_almost_equal(ar.dep_params, dep_params_true[gsize - 1])
        assert_almost_equal(mdf.params, params_true[gsize - 1])