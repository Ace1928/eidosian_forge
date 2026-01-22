from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def test_invalid_standardize_1d():
    endog = np.zeros(100) + 10
    endog_pd = pd.Series(endog, name='y1')
    options = [([], 10), (10, []), ([], []), ([1, 2], [1.0]), ([1], [1, 2.0])]
    msg = 'Invalid value passed for `standardize`: each element must be shaped'
    for standardize in options:
        with pytest.raises(ValueError, match=msg):
            dynamic_factor_mq.DynamicFactorMQ(endog, factors=1, factor_orders=1, idiosyncratic_ar1=False, standardize=standardize)
    options = [(pd.Series(10), pd.Series(10)), (pd.Series(10, index=['y']), pd.Series(10, index=['y1'])), (pd.Series(10, index=['y1']), pd.Series(10, index=['y1'])), (pd.Series([10], index=['y']), pd.Series([10, 1], index=['y1', 'y2']))]
    msg = 'Invalid value passed for `standardize`: if a Pandas Series, must have index'
    for standardize in options:
        with pytest.raises(ValueError, match=msg):
            dynamic_factor_mq.DynamicFactorMQ(endog, factors=1, factor_orders=1, idiosyncratic_ar1=False, standardize=standardize)
    options = [(pd.Series(10), pd.Series(10)), (pd.Series(10, index=['y']), pd.Series(10, index=['y1'])), (pd.Series(10, index=['y']), pd.Series(10, index=['y'])), (pd.Series([10], index=['y']), pd.Series([10, 1], index=['y1', 'y2']))]
    msg = 'Invalid value passed for `standardize`: if a Pandas Series, must have index'
    for standardize in options:
        with pytest.raises(ValueError, match=msg):
            dynamic_factor_mq.DynamicFactorMQ(endog_pd, factors=1, factor_orders=1, idiosyncratic_ar1=False, standardize=standardize)