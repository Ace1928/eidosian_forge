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
def test_noconstant():
    exog = exog_st[:, :-1]
    mod = gmm.IV2SLS(endog, exog, instrument)
    res = mod.fit()
    assert_equal(res.fvalue, np.nan)
    summ = res.summary()
    assert_equal(len(summ.tables[1]), len(res.params) + 1)