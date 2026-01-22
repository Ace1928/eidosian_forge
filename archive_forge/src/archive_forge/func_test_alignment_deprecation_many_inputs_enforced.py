from functools import partial
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_extension_array_dtype
def test_alignment_deprecation_many_inputs_enforced():
    numba = pytest.importorskip('numba')

    @numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
    def my_ufunc(x, y, z):
        return x + y + z
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = pd.DataFrame({'b': [1, 2, 3], 'c': [4, 5, 6]})
    df3 = pd.DataFrame({'a': [1, 2, 3], 'c': [4, 5, 6]})
    result = my_ufunc(df1, df2, df3)
    expected = pd.DataFrame(np.full((3, 3), np.nan), columns=['a', 'b', 'c'])
    tm.assert_frame_equal(result, expected)
    with tm.assert_produces_warning(None):
        result = my_ufunc(df1, df1, df1)
    expected = pd.DataFrame([[3.0, 12.0], [6.0, 15.0], [9.0, 18.0]], columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)
    msg = 'operands could not be broadcast together with shapes \\(3,3\\) \\(3,3\\) \\(3,2\\)'
    with pytest.raises(ValueError, match=msg):
        my_ufunc(df1, df2, df3.values)
    with tm.assert_produces_warning(None):
        result = my_ufunc(df1, df2.values, df3.values)
    tm.assert_frame_equal(result, expected)
    msg = 'operands could not be broadcast together with shapes \\(3,2\\) \\(3,3\\) \\(3,3\\)'
    with pytest.raises(ValueError, match=msg):
        my_ufunc(df1.values, df2, df3)