import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', ['int64', 'float64'], ids=['single-block', 'mixed-block'])
def test_subset_set_column_with_loc(backend, using_copy_on_write, warn_copy_on_write, using_array_manager, dtype):
    _, DataFrame, _ = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': np.array([7, 8, 9], dtype=dtype)})
    df_orig = df.copy()
    subset = df[1:3]
    if using_copy_on_write:
        subset.loc[:, 'a'] = np.array([10, 11], dtype='int64')
    elif warn_copy_on_write:
        with tm.assert_cow_warning():
            subset.loc[:, 'a'] = np.array([10, 11], dtype='int64')
    else:
        with pd.option_context('chained_assignment', 'warn'):
            with tm.assert_produces_warning(None, raise_on_extra_warnings=not using_array_manager):
                subset.loc[:, 'a'] = np.array([10, 11], dtype='int64')
    subset._mgr._verify_integrity()
    expected = DataFrame({'a': [10, 11], 'b': [5, 6], 'c': np.array([8, 9], dtype=dtype)}, index=range(1, 3))
    tm.assert_frame_equal(subset, expected)
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        df_orig.loc[1:3, 'a'] = np.array([10, 11], dtype='int64')
        tm.assert_frame_equal(df, df_orig)