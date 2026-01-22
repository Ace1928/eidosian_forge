import re
import unicodedata
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_get_dummies_basic_drop_first(self, sparse):
    s_list = list('abc')
    s_series = Series(s_list)
    s_series_index = Series(s_list, list('ABC'))
    expected = DataFrame({'b': [0, 1, 0], 'c': [0, 0, 1]}, dtype=bool)
    result = get_dummies(s_list, drop_first=True, sparse=sparse)
    if sparse:
        expected = expected.apply(SparseArray, fill_value=False)
    tm.assert_frame_equal(result, expected)
    result = get_dummies(s_series, drop_first=True, sparse=sparse)
    tm.assert_frame_equal(result, expected)
    expected.index = list('ABC')
    result = get_dummies(s_series_index, drop_first=True, sparse=sparse)
    tm.assert_frame_equal(result, expected)