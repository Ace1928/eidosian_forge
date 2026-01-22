import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_ufunc_numeric():
    arr = pd.array([True, False, None], dtype='boolean')
    res = np.sqrt(arr)
    expected = pd.array([1, 0, None], dtype='Float32')
    tm.assert_extension_array_equal(res, expected)