import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('func, expected', [['mean', DataFrame({0: range(5), 1: range(4, 9), 2: [7.428571, 9, 10.571429, 12.142857, 13.714286]}, dtype=float)], ['std', DataFrame({0: [np.nan] * 5, 1: [4.242641] * 5, 2: [4.6291, 5.196152, 5.781745, 6.380775, 6.989788]})], ['var', DataFrame({0: [np.nan] * 5, 1: [18.0] * 5, 2: [21.428571, 27, 33.428571, 40.714286, 48.857143]})]])
def test_float_dtype_ewma(func, expected, float_numpy_dtype):
    df = DataFrame({0: range(5), 1: range(6, 11), 2: range(10, 20, 2)}, dtype=float_numpy_dtype)
    msg = 'Support for axis=1 in DataFrame.ewm is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        e = df.ewm(alpha=0.5, axis=1)
    result = getattr(e, func)()
    tm.assert_frame_equal(result, expected)