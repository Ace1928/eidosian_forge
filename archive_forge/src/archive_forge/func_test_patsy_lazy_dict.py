from statsmodels.compat.pandas import assert_series_equal
from io import StringIO
import warnings
import numpy as np
import numpy.testing as npt
import pandas as pd
import patsy
import pytest
from statsmodels.datasets import cpunish
from statsmodels.datasets.longley import load, load_pandas
from statsmodels.formula.api import ols
from statsmodels.formula.formulatools import make_hypotheses_matrices
from statsmodels.tools import add_constant
from statsmodels.tools.testing import assert_equal
def test_patsy_lazy_dict():

    class LazyDict(dict):

        def __init__(self, data):
            self.data = data

        def __missing__(self, key):
            return np.array(self.data[key])
    data = cpunish.load_pandas().data
    data = LazyDict(data)
    res = ols('EXECUTIONS ~ SOUTH + INCOME', data=data).fit()
    res2 = res.predict(data)
    npt.assert_allclose(res.fittedvalues, res2)
    data = cpunish.load_pandas().data
    data.loc[0, 'INCOME'] = np.nan
    data = LazyDict(data)
    data.index = cpunish.load_pandas().data.index
    res = ols('EXECUTIONS ~ SOUTH + INCOME', data=data).fit()
    res2 = res.predict(data)
    assert_equal(res.fittedvalues, res2)
    assert_equal(len(res2) + 1, len(cpunish.load_pandas().data))