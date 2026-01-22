import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('subtype', ['int64', 'uint64'])
def test_subtype_integer_with_non_integer_borders(self, subtype):
    index = interval_range(0.0, 3.0, freq=0.25)
    dtype = IntervalDtype(subtype, 'right')
    result = index.astype(dtype)
    expected = IntervalIndex.from_arrays(index.left.astype(subtype), index.right.astype(subtype), closed=index.closed)
    tm.assert_index_equal(result, expected)