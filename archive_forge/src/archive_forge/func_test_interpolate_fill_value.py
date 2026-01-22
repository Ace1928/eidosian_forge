import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interpolate_fill_value(self):
    pytest.importorskip('scipy')
    ser = Series([np.nan, 0, 1, np.nan, 3, np.nan])
    result = ser.interpolate(method='nearest', fill_value=0)
    expected = Series([np.nan, 0, 1, 1, 3, 0])
    tm.assert_series_equal(result, expected)