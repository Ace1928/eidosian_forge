import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_spline(self):
    pytest.importorskip('scipy')
    s = Series([1, 2, np.nan, 4, 5, np.nan, 7])
    result = s.interpolate(method='spline', order=1)
    expected = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    tm.assert_series_equal(result, expected)