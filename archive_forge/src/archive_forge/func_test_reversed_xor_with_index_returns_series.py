from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_reversed_xor_with_index_returns_series(self):
    ser = Series([True, True, False, False])
    idx1 = Index([True, False, True, False], dtype=bool)
    idx2 = Index([1, 0, 1, 0])
    expected = Series([False, True, True, False])
    result = idx1 ^ ser
    tm.assert_series_equal(result, expected)
    result = idx2 ^ ser
    tm.assert_series_equal(result, expected)