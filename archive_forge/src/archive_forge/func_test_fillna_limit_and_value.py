import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_limit_and_value(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)))
    df.iloc[2:7, 0] = np.nan
    df.iloc[3:5, 2] = np.nan
    expected = df.copy()
    expected.iloc[2, 0] = 999
    expected.iloc[3, 2] = 999
    result = df.fillna(999, limit=1)
    tm.assert_frame_equal(result, expected)