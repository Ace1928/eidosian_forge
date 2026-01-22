import decimal
import numpy as np
import pytest
from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas import (
import pandas._testing as tm
def test_downcast_conversion_empty(any_real_numpy_dtype):
    dtype = any_real_numpy_dtype
    arr = np.array([], dtype=dtype)
    result = maybe_downcast_to_dtype(arr, np.dtype('int64'))
    tm.assert_numpy_array_equal(result, np.array([], dtype=np.int64))