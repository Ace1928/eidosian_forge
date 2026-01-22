import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interp_inplace(self, using_copy_on_write):
    df = DataFrame({'a': [1.0, 2.0, np.nan, 4.0]})
    expected = DataFrame({'a': [1.0, 2.0, 3.0, 4.0]})
    expected_cow = df.copy()
    result = df.copy()
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            return_value = result['a'].interpolate(inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected_cow)
    else:
        with tm.assert_produces_warning(FutureWarning, match='inplace method'):
            return_value = result['a'].interpolate(inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)
    result = df.copy()
    msg = "The 'downcast' keyword in Series.interpolate is deprecated"
    if using_copy_on_write:
        with tm.assert_produces_warning((FutureWarning, ChainedAssignmentError), match=msg):
            return_value = result['a'].interpolate(inplace=True, downcast='infer')
        assert return_value is None
        tm.assert_frame_equal(result, expected_cow)
    else:
        with tm.assert_produces_warning(FutureWarning, match=msg):
            return_value = result['a'].interpolate(inplace=True, downcast='infer')
        assert return_value is None
        tm.assert_frame_equal(result, expected.astype('int64'))