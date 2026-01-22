from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_truncate_out_of_bounds(self, frame_or_series):
    shape = [2000] + [1] * (frame_or_series._AXIS_LEN - 1)
    small = construct(frame_or_series, shape, dtype='int8', value=1)
    tm.assert_equal(small.truncate(), small)
    tm.assert_equal(small.truncate(before=0, after=3000.0), small)
    tm.assert_equal(small.truncate(before=-1, after=2000.0), small)
    shape = [2000000] + [1] * (frame_or_series._AXIS_LEN - 1)
    big = construct(frame_or_series, shape, dtype='int8', value=1)
    tm.assert_equal(big.truncate(), big)
    tm.assert_equal(big.truncate(before=0, after=3000000.0), big)
    tm.assert_equal(big.truncate(before=-1, after=2000000.0), big)