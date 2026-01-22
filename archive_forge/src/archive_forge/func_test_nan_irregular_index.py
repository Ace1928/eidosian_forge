import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nan_irregular_index(self):
    s = Series([1, 2, np.nan, 4], index=[1, 3, 5, 9])
    result = s.interpolate()
    expected = Series([1.0, 2.0, 3.0, 4.0], index=[1, 3, 5, 9])
    tm.assert_series_equal(result, expected)