from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
def test_boolean_comparison(self):
    df = DataFrame(np.arange(6).reshape((3, 2)))
    b = np.array([2, 2])
    b_r = np.atleast_2d([2, 2])
    b_c = b_r.T
    lst = [2, 2, 2]
    tup = tuple(lst)
    expected = DataFrame([[False, False], [False, True], [True, True]])
    result = df > b
    tm.assert_frame_equal(result, expected)
    result = df.values > b
    tm.assert_numpy_array_equal(result, expected.values)
    msg1d = 'Unable to coerce to Series, length must be 2: given 3'
    msg2d = 'Unable to coerce to DataFrame, shape must be'
    msg2db = 'operands could not be broadcast together with shapes'
    with pytest.raises(ValueError, match=msg1d):
        df > lst
    with pytest.raises(ValueError, match=msg1d):
        df > tup
    result = df > b_r
    tm.assert_frame_equal(result, expected)
    result = df.values > b_r
    tm.assert_numpy_array_equal(result, expected.values)
    with pytest.raises(ValueError, match=msg2d):
        df > b_c
    with pytest.raises(ValueError, match=msg2db):
        df.values > b_c
    expected = DataFrame([[False, False], [True, False], [False, False]])
    result = df == b
    tm.assert_frame_equal(result, expected)
    with pytest.raises(ValueError, match=msg1d):
        df == lst
    with pytest.raises(ValueError, match=msg1d):
        df == tup
    result = df == b_r
    tm.assert_frame_equal(result, expected)
    result = df.values == b_r
    tm.assert_numpy_array_equal(result, expected.values)
    with pytest.raises(ValueError, match=msg2d):
        df == b_c
    assert df.values.shape != b_c.shape
    df = DataFrame(np.arange(6).reshape((3, 2)), columns=list('AB'), index=list('abc'))
    expected.index = df.index
    expected.columns = df.columns
    with pytest.raises(ValueError, match=msg1d):
        df == lst
    with pytest.raises(ValueError, match=msg1d):
        df == tup