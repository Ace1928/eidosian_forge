import pandas as pd
import pandas._testing as tm
def test_array_setitem():
    arr = pd.Series([1, 2], dtype='Int64').array
    arr[arr > 1] = 1
    expected = pd.array([1, 1], dtype='Int64')
    tm.assert_extension_array_equal(arr, expected)