import warnings
from statsmodels.compat.pandas import PD_LT_1_4
import os
import numpy as np
import pandas as pd
from statsmodels.multivariate.factor import Factor
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
def test_factor_missing():
    xm = X.iloc[:, 1:-1].copy()
    nobs, k_endog = xm.shape
    xm.iloc[2, 2] = np.nan
    mod = Factor(xm, 2)
    assert_equal(mod.nobs, nobs - 1)
    assert_equal(mod.k_endog, k_endog)
    assert_equal(mod.endog.shape, (nobs - 1, k_endog))