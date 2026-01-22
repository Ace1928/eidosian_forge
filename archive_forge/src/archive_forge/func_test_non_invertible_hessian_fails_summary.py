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
def test_non_invertible_hessian_fails_summary():
    data = cpunish.load_pandas()
    data.endog[:] = 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mod = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
        res = mod.fit(maxiter=1, method='bfgs', max_start_irls=0)
        res.summary()