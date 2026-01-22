import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('subtype', ['float64', 'datetime64[ns]', 'timedelta64[ns]'])
def test_subtype_conversion(self, index, subtype):
    dtype = IntervalDtype(subtype, index.closed)
    result = index.astype(dtype)
    expected = IntervalIndex.from_arrays(index.left.astype(subtype), index.right.astype(subtype), closed=index.closed)
    tm.assert_index_equal(result, expected)