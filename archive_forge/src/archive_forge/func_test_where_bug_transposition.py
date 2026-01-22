from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_bug_transposition(self):
    a = DataFrame({0: [1, 2], 1: [3, 4], 2: [5, 6]})
    b = DataFrame({0: [np.nan, 8], 1: [9, np.nan], 2: [np.nan, np.nan]})
    do_not_replace = b.isna() | (a > b)
    expected = a.copy()
    expected[~do_not_replace] = b
    msg = "Downcasting behavior in Series and DataFrame methods 'where'"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = a.where(do_not_replace, b)
    tm.assert_frame_equal(result, expected)
    a = DataFrame({0: [4, 6], 1: [1, 0]})
    b = DataFrame({0: [np.nan, 3], 1: [3, np.nan]})
    do_not_replace = b.isna() | (a > b)
    expected = a.copy()
    expected[~do_not_replace] = b
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = a.where(do_not_replace, b)
    tm.assert_frame_equal(result, expected)