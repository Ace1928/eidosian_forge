import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nan_str_index(self):
    s = Series([0, 1, 2, np.nan], index=list('abcd'))
    result = s.interpolate()
    expected = Series([0.0, 1.0, 2.0, 2.0], index=list('abcd'))
    tm.assert_series_equal(result, expected)