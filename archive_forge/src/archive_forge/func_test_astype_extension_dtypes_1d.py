import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['Int64', 'Int32', 'Int16'])
def test_astype_extension_dtypes_1d(self, dtype):
    df = DataFrame({'a': [1.0, 2.0, 3.0]})
    expected1 = DataFrame({'a': pd.array([1, 2, 3], dtype=dtype)})
    tm.assert_frame_equal(df.astype(dtype), expected1)
    tm.assert_frame_equal(df.astype('int64').astype(dtype), expected1)
    df = DataFrame({'a': [1.0, 2.0, 3.0]})
    df['a'] = df['a'].astype(dtype)
    expected2 = DataFrame({'a': pd.array([1, 2, 3], dtype=dtype)})
    tm.assert_frame_equal(df, expected2)
    tm.assert_frame_equal(df.astype(dtype), expected1)
    tm.assert_frame_equal(df.astype('int64').astype(dtype), expected1)