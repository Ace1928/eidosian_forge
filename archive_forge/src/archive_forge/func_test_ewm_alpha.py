import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewm_alpha():
    arr = np.random.default_rng(2).standard_normal(100)
    locs = np.arange(20, 40)
    arr[locs] = np.nan
    s = Series(arr)
    a = s.ewm(alpha=0.6172269988916967).mean()
    b = s.ewm(com=0.6201494778997305).mean()
    c = s.ewm(span=2.240298955799461).mean()
    d = s.ewm(halflife=0.721792864318).mean()
    tm.assert_series_equal(a, b)
    tm.assert_series_equal(a, c)
    tm.assert_series_equal(a, d)