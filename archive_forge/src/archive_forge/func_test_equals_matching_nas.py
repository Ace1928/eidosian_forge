from contextlib import nullcontext
import copy
import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import is_float
from pandas import (
import pandas._testing as tm
def test_equals_matching_nas():
    left = Series([np.datetime64('NaT')], dtype=object)
    right = Series([np.datetime64('NaT')], dtype=object)
    assert left.equals(right)
    with tm.assert_produces_warning(FutureWarning, match='Dtype inference'):
        assert Index(left).equals(Index(right))
    assert left.array.equals(right.array)
    left = Series([np.timedelta64('NaT')], dtype=object)
    right = Series([np.timedelta64('NaT')], dtype=object)
    assert left.equals(right)
    with tm.assert_produces_warning(FutureWarning, match='Dtype inference'):
        assert Index(left).equals(Index(right))
    assert left.array.equals(right.array)
    left = Series([np.float64('NaN')], dtype=object)
    right = Series([np.float64('NaN')], dtype=object)
    assert left.equals(right)
    assert Index(left, dtype=left.dtype).equals(Index(right, dtype=right.dtype))
    assert left.array.equals(right.array)