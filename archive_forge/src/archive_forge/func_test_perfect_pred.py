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
def test_perfect_pred(iris):
    y = iris[:, -1]
    X = iris[:, :-1]
    X = X[y != 2]
    y = y[y != 2]
    X = add_constant(X, prepend=True)
    glm = GLM(y, X, family=sm.families.Binomial())
    with pytest.warns(PerfectSeparationWarning):
        glm.fit()