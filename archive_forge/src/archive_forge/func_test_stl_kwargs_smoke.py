from pathlib import Path
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import MSTL
def test_stl_kwargs_smoke(data):
    stl_kwargs = {'period': 12, 'seasonal': 15, 'trend': 17, 'low_pass': 15, 'seasonal_deg': 0, 'trend_deg': 1, 'low_pass_deg': 1, 'seasonal_jump': 2, 'trend_jump': 2, 'low_pass_jump': 3, 'robust': False, 'inner_iter': 3, 'outer_iter': 3}
    periods = (5, 6, 7)
    mod = MSTL(endog=data, periods=periods, lmbda='auto', stl_kwargs=stl_kwargs)
    mod.fit()