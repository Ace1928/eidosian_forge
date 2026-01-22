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
def test_offset_exposure(self):
    np.random.seed(382304)
    endog = np.random.randint(0, 10, 100)
    exog = np.random.normal(size=(100, 3))
    exposure = np.random.uniform(1, 2, 100)
    offset = np.random.uniform(1, 2, 100)
    mod1 = GLM(endog, exog, family=sm.families.Poisson(), offset=offset, exposure=exposure).fit()
    offset2 = offset + np.log(exposure)
    mod2 = GLM(endog, exog, family=sm.families.Poisson(), offset=offset2).fit()
    assert_almost_equal(mod1.params, mod2.params)
    assert_allclose(mod1.null, mod2.null, rtol=1e-10)
    mod1_ = mod1.model
    kwds = mod1_._get_init_kwds()
    assert_allclose(kwds['exposure'], exposure, rtol=1e-14)
    assert_allclose(kwds['offset'], mod1_.offset, rtol=1e-14)
    mod3 = mod1_.__class__(mod1_.endog, mod1_.exog, **kwds)
    assert_allclose(mod3.exposure, mod1_.exposure, rtol=1e-14)
    assert_allclose(mod3.offset, mod1_.offset, rtol=1e-14)
    resr1 = mod1.model.fit_regularized()
    resr2 = mod2.model.fit_regularized()
    assert_allclose(resr1.params, resr2.params, rtol=1e-10)