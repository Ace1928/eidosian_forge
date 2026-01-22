import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_concat_series_axis1(self):
    ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
    pieces = [ts[:-2], ts[2:], ts[2:-2]]
    result = concat(pieces, axis=1)
    expected = DataFrame(pieces).T
    tm.assert_frame_equal(result, expected)
    result = concat(pieces, keys=['A', 'B', 'C'], axis=1)
    expected = DataFrame(pieces, index=['A', 'B', 'C']).T
    tm.assert_frame_equal(result, expected)