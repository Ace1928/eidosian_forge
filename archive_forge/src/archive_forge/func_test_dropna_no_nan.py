import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_dropna_no_nan(self):
    for ser in [Series([1, 2, 3], name='x'), Series([False, True, False], name='x')]:
        result = ser.dropna()
        tm.assert_series_equal(result, ser)
        assert result is not ser
        s2 = ser.copy()
        return_value = s2.dropna(inplace=True)
        assert return_value is None
        tm.assert_series_equal(s2, ser)