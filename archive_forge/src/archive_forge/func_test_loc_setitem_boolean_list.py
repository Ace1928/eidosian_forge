from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
@pytest.mark.parametrize('indexing_func', [list, np.array])
@pytest.mark.parametrize('rhs_func', [list, np.array])
def test_loc_setitem_boolean_list(self, rhs_func, indexing_func):
    ser = Series([0, 1, 2])
    ser.iloc[indexing_func([True, False, True])] = rhs_func([5, 10])
    expected = Series([5, 1, 10])
    tm.assert_series_equal(ser, expected)
    df = DataFrame({'a': [0, 1, 2]})
    df.iloc[indexing_func([True, False, True])] = rhs_func([[5], [10]])
    expected = DataFrame({'a': [5, 1, 10]})
    tm.assert_frame_equal(df, expected)