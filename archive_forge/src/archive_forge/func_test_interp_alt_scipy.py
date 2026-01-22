import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interp_alt_scipy(self):
    pytest.importorskip('scipy')
    df = DataFrame({'A': [1, 2, np.nan, 4, 5, np.nan, 7], 'C': [1, 2, 3, 5, 8, 13, 21]})
    result = df.interpolate(method='barycentric')
    expected = df.copy()
    expected.loc[2, 'A'] = 3
    expected.loc[5, 'A'] = 6
    tm.assert_frame_equal(result, expected)
    msg = "The 'downcast' keyword in DataFrame.interpolate is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.interpolate(method='barycentric', downcast='infer')
    tm.assert_frame_equal(result, expected.astype(np.int64))
    result = df.interpolate(method='krogh')
    expectedk = df.copy()
    expectedk['A'] = expected['A']
    tm.assert_frame_equal(result, expectedk)
    result = df.interpolate(method='pchip')
    expected.loc[2, 'A'] = 3
    expected.loc[5, 'A'] = 6.0
    tm.assert_frame_equal(result, expected)