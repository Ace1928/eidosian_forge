from pathlib import Path
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import MSTL
@pytest.mark.parametrize('data, lmbda', [(data, 0.1), (data, 1), (data, -3.0), (data, 'auto')], indirect=['data'])
def test_fit_with_box_cox(data, lmbda):
    periods = (5, 6, 7)
    mod = MSTL(endog=data, periods=periods, lmbda=lmbda)
    mod.fit()