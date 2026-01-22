import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_inplace_with_columns_limit_and_value(self):
    df = DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5], [np.nan, 3, np.nan, 4]], columns=list('ABCD'))
    expected = df.fillna(axis=1, value=100, limit=1)
    assert expected is not df
    df.fillna(axis=1, value=100, limit=1, inplace=True)
    tm.assert_frame_equal(df, expected)