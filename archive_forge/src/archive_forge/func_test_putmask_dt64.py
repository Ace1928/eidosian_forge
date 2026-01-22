import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', ['US/Pacific', None])
def test_putmask_dt64(self, tz):
    dti = date_range('2016-01-01', periods=9, tz=tz)
    idx = IntervalIndex.from_breaks(dti)
    mask = np.zeros(idx.shape, dtype=bool)
    mask[0:3] = True
    result = idx.putmask(mask, idx[-1])
    expected = IntervalIndex([idx[-1]] * 3 + list(idx[3:]))
    tm.assert_index_equal(result, expected)