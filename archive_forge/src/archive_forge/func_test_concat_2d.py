import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
def test_concat_2d(self, data):
    left = type(data)._concat_same_type([data, data]).reshape(-1, 2)
    right = left.copy()
    result = left._concat_same_type([left, right], axis=0)
    expected = data._concat_same_type([data] * 4).reshape(-1, 2)
    tm.assert_extension_array_equal(result, expected)
    result = left._concat_same_type([left, right], axis=1)
    assert result.shape == (len(data), 4)
    tm.assert_extension_array_equal(result[:, :2], left)
    tm.assert_extension_array_equal(result[:, 2:], right)
    msg = 'axis 2 is out of bounds for array of dimension 2'
    with pytest.raises(ValueError, match=msg):
        left._concat_same_type([left, right], axis=2)