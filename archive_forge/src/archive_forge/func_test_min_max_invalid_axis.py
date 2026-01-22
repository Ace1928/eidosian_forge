import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_min_max_invalid_axis(self, left_right_dtypes):
    left, right = left_right_dtypes
    left = left.copy(deep=True)
    right = right.copy(deep=True)
    arr = IntervalArray.from_arrays(left, right)
    msg = '`axis` must be fewer than the number of dimensions'
    for axis in [-2, 1]:
        with pytest.raises(ValueError, match=msg):
            arr.min(axis=axis)
        with pytest.raises(ValueError, match=msg):
            arr.max(axis=axis)
    msg = "'>=' not supported between"
    with pytest.raises(TypeError, match=msg):
        arr.min(axis='foo')
    with pytest.raises(TypeError, match=msg):
        arr.max(axis='foo')