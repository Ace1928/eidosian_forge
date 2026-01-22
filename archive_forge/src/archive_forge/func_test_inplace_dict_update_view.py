import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
@td.skip_array_manager_invalid_test
@pytest.mark.parametrize('val', [-1, {'x': -1, 'y': -1}])
def test_inplace_dict_update_view(self, val, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'x': [np.nan, 2], 'y': [np.nan, 2]})
    df_orig = df.copy()
    result_view = df[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        df.fillna(val, inplace=True)
    expected = DataFrame({'x': [-1, 2.0], 'y': [-1.0, 2]})
    tm.assert_frame_equal(df, expected)
    if using_copy_on_write:
        tm.assert_frame_equal(result_view, df_orig)
    else:
        tm.assert_frame_equal(result_view, expected)