import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_spline_interpolation(self):
    pytest.importorskip('scipy')
    s = Series(np.arange(10) ** 2, dtype='float')
    s[np.random.default_rng(2).integers(0, 9, 3)] = np.nan
    result1 = s.interpolate(method='spline', order=1)
    expected1 = s.interpolate(method='spline', order=1)
    tm.assert_series_equal(result1, expected1)