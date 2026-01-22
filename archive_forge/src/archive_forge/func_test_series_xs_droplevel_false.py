import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_series_xs_droplevel_false(self):
    mi = MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'x')], names=['level1', 'level2'])
    ser = Series([1, 1, 1], index=mi)
    result = ser.xs('a', axis=0, drop_level=False)
    expected = Series([1, 1], index=MultiIndex.from_tuples([('a', 'x'), ('a', 'y')], names=['level1', 'level2']))
    tm.assert_series_equal(result, expected)