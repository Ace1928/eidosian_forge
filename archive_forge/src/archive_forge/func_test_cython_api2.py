import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
def test_cython_api2():
    df = DataFrame([[1, 2, np.nan], [1, np.nan, 9], [3, 4, 9]], columns=['A', 'B', 'C'])
    expected = DataFrame([[2, np.nan], [np.nan, 9], [4, 9]], columns=['B', 'C'])
    result = df.groupby('A').cumsum()
    tm.assert_frame_equal(result, expected)
    result = df.groupby('A', as_index=False).cumsum()
    tm.assert_frame_equal(result, expected)
    result = df.groupby('A').cumsum(axis=1)
    expected = df.cumsum(axis=1)
    tm.assert_frame_equal(result, expected)
    result = df.groupby('A').cumprod(axis=1)
    expected = df.cumprod(axis=1)
    tm.assert_frame_equal(result, expected)