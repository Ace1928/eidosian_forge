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
def test_gmm_basic():
    cd = np.array([1.5, 1.5, 1.7, 2.2, 2.0, 1.8, 1.8, 2.2, 1.9, 1.6, 1.8, 2.2, 2.0, 1.5, 1.1, 1.5, 1.4, 1.7, 1.42, 1.9])
    dcd = np.array([0, 0.2, 0.5, -0.2, -0.2, 0, 0.4, -0.3, -0.3, 0.2, 0.4, -0.2, -0.5, -0.4, 0.4, -0.1, 0.3, -0.28, 0.48, 0.2])
    inst = np.column_stack((np.ones(len(cd)), cd))

    class GMMbase(gmm.GMM):

        def momcond(self, params):
            p0, p1, p2, p3 = params
            endog = self.endog[:, None]
            exog = self.exog
            inst = self.instrument
            mom0 = (endog - p0 - p1 * exog) * inst
            mom1 = ((endog - p0 - p1 * exog) ** 2 - p2 * exog ** (2 * p3) / 12) * inst
            g = np.column_stack((mom0, mom1))
            return g
    beta0 = np.array([0.1, 0.1, 0.01, 1])
    res = GMMbase(endog=dcd, exog=cd, instrument=inst, k_moms=4, k_params=4).fit(beta0, optim_args={'disp': 0})
    summ = res.summary()
    assert_equal(len(summ.tables[1]), len(res.params) + 1)
    pnames = ['p%2d' % i for i in range(len(res.params))]
    assert_equal(res.model.exog_names, pnames)
    mod = GMMbase(endog=dcd, exog=cd, instrument=inst, k_moms=4, k_params=4)
    pnames = ['beta', 'gamma', 'psi', 'phi']
    mod.set_param_names(pnames)
    res1 = mod.fit(beta0, optim_args={'disp': 0})
    assert_equal(res1.model.exog_names, pnames)