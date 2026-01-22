import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('subtype', ['int64', 'uint64'])
def test_subtype_integer(self, index, subtype):
    dtype = IntervalDtype(subtype, 'right')
    if subtype != 'int64':
        msg = 'Cannot convert interval\\[(timedelta64|datetime64)\\[ns.*\\], .*\\] to interval\\[uint64, .*\\]'
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)
        return
    result = index.astype(dtype)
    new_left = index.left.astype(subtype)
    new_right = index.right.astype(subtype)
    expected = IntervalIndex.from_arrays(new_left, new_right, closed=index.closed)
    tm.assert_index_equal(result, expected)