import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.tests.arrays.masked_shared import (
def test_equals_nan_vs_na():
    mask = np.zeros(3, dtype=bool)
    data = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    left = FloatingArray(data, mask)
    assert left.equals(left)
    tm.assert_extension_array_equal(left, left)
    assert left.equals(left.copy())
    assert left.equals(FloatingArray(data.copy(), mask.copy()))
    mask2 = np.array([False, True, False], dtype=bool)
    data2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    right = FloatingArray(data2, mask2)
    assert right.equals(right)
    tm.assert_extension_array_equal(right, right)
    assert not left.equals(right)
    mask[1] = True
    assert left.equals(right)