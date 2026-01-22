import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.parametrize('func', ['cov', 'corr'])
def test_rolling_pairwise_cov_corr(func, frame):
    result = getattr(frame.rolling(window=10, min_periods=5), func)()
    result = result.loc[(slice(None), 1), 5]
    result.index = result.index.droplevel(1)
    expected = getattr(frame[1].rolling(window=10, min_periods=5), func)(frame[5])
    tm.assert_series_equal(result, expected, check_names=False)