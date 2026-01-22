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
def test_qaic():
    import patsy
    ldose = np.concatenate((np.arange(6), np.arange(6)))
    sex = ['M'] * 6 + ['F'] * 6
    numdead = [10, 4, 9, 12, 18, 20, 0, 2, 6, 10, 12, 16]
    df = pd.DataFrame({'ldose': ldose, 'sex': sex, 'numdead': numdead})
    df['numalive'] = 20 - df['numdead']
    df['SF'] = df['numdead']
    y = df[['numalive', 'numdead']].values
    x = patsy.dmatrix('sex*ldose', data=df, return_type='dataframe')
    m = GLM(y, x, family=sm.families.Binomial())
    r = m.fit()
    scale = 2.412699
    qaic = r.info_criteria(crit='qaic', scale=scale)
    assert_allclose(qaic, 29.13266, rtol=1e-05, atol=1e-05)
    qaic1 = r.info_criteria(crit='qaic', scale=scale, dk_params=1)
    assert_allclose(qaic1, 31.13266, rtol=1e-05, atol=1e-05)