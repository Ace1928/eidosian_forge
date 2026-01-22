import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.base import BaseOpsUtil
def test_compare_to_string(self, dtype):
    ser = pd.Series([1, None], dtype=dtype)
    result = ser == 'a'
    expected = pd.Series([False, pd.NA], dtype='boolean')
    tm.assert_series_equal(result, expected)