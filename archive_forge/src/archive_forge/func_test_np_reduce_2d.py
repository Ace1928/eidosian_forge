import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
def test_np_reduce_2d():
    raw = np.arange(12).reshape(4, 3)
    arr = NumpyExtensionArray(raw)
    res = np.maximum.reduce(arr, axis=0)
    tm.assert_extension_array_equal(res, arr[-1])
    alt = arr.max(axis=0)
    tm.assert_extension_array_equal(alt, arr[-1])