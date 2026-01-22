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
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64])
def test_int_exog(dtype):
    count1, n1, count2, n2 = (60, 51477.5, 30, 54308.7)
    y = [count1, count2]
    x = np.asarray([[1, 1], [1, 0]]).astype(dtype)
    exposure = np.asarray([n1, n2])
    mod = GLM(y, x, exposure=exposure, family=sm.families.Poisson())
    res = mod.fit(method='bfgs', max_start_irls=0)
    assert isinstance(res.params, np.ndarray)