from functools import partial
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_extension_array_dtype
@pytest.mark.parametrize('func,arg,expected', [(np.add, 1, [2, 3, 4, 5]), (partial(np.add, where=[[False, True], [True, False]]), np.array([[1, 1], [1, 1]]), [0, 3, 4, 0]), (np.power, np.array([[1, 1], [2, 2]]), [1, 2, 9, 16]), (np.subtract, 2, [-1, 0, 1, 2]), (partial(np.negative, where=np.array([[False, True], [True, False]])), None, [0, -2, -3, 0])])
def test_ufunc_passes_args(func, arg, expected):
    arr = np.array([[1, 2], [3, 4]])
    df = pd.DataFrame(arr)
    result_inplace = np.zeros_like(arr)
    if arg is None:
        result = func(df, out=result_inplace)
    else:
        result = func(df, arg, out=result_inplace)
    expected = np.array(expected).reshape(2, 2)
    tm.assert_numpy_array_equal(result_inplace, expected)
    expected = pd.DataFrame(expected)
    tm.assert_frame_equal(result, expected)