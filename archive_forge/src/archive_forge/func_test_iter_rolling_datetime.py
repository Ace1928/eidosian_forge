from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('expected,expected_index,window', [([[0], [1], [2], [3], [4]], [date_range('2020-01-01', periods=1, freq='D'), date_range('2020-01-02', periods=1, freq='D'), date_range('2020-01-03', periods=1, freq='D'), date_range('2020-01-04', periods=1, freq='D'), date_range('2020-01-05', periods=1, freq='D')], '1D'), ([[0], [0, 1], [1, 2], [2, 3], [3, 4]], [date_range('2020-01-01', periods=1, freq='D'), date_range('2020-01-01', periods=2, freq='D'), date_range('2020-01-02', periods=2, freq='D'), date_range('2020-01-03', periods=2, freq='D'), date_range('2020-01-04', periods=2, freq='D')], '2D'), ([[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]], [date_range('2020-01-01', periods=1, freq='D'), date_range('2020-01-01', periods=2, freq='D'), date_range('2020-01-01', periods=3, freq='D'), date_range('2020-01-02', periods=3, freq='D'), date_range('2020-01-03', periods=3, freq='D')], '3D')])
def test_iter_rolling_datetime(expected, expected_index, window):
    ser = Series(range(5), index=date_range(start='2020-01-01', periods=5, freq='D'))
    expected = [Series(values, index=idx) for values, idx in zip(expected, expected_index)]
    for expected, actual in zip(expected, ser.rolling(window)):
        tm.assert_series_equal(actual, expected)