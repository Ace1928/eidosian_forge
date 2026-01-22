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
def test_tweedie_elastic_net():
    p = 1.5
    y, x = gen_tweedie(p)
    fam = sm.families.Tweedie(var_power=p, eql=True)
    model1 = sm.GLM(y, x, family=fam)
    nnz = []
    for alpha in np.linspace(0, 10, 20):
        result1 = model1.fit_regularized(L1_wt=0.5, alpha=alpha)
        nnz.append((np.abs(result1.params) > 0).sum())
    nnz = np.unique(nnz)
    assert len(nnz) == 5