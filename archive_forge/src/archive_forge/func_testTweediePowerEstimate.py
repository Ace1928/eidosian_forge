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
def testTweediePowerEstimate():
    data = cpunish.load_pandas()
    y = [100113.835, 6896.68315, 6157.26842, 1417.18806, 511.776456, 255.369154, 10.7147443, 3.56874698, 0.0406797842, 7.06996731e-05, 2.10165106e-07, 4.34276938e-08, 1.5635404e-09, 0.0, 0.0, 0.0, 0.0]
    model1 = sm.GLM(y, data.exog[['INCOME', 'SOUTH']], family=sm.families.Tweedie(link=sm.families.links.Log(), var_power=1.5))
    res1 = model1.fit()
    model2 = sm.GLM((y - res1.mu) ** 2, np.column_stack((np.ones(len(res1.mu)), np.log(res1.mu))), family=sm.families.Gamma(sm.families.links.Log()))
    res2 = model2.fit()
    p = model1.estimate_tweedie_power(res1.mu)
    assert_allclose(p, res2.params[1], rtol=0.25)