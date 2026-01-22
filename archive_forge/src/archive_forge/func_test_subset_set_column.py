import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_subset_set_column(backend, using_copy_on_write, warn_copy_on_write):
    dtype_backend, DataFrame, _ = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    subset = df[1:3]
    if dtype_backend == 'numpy':
        arr = np.array([10, 11], dtype='int64')
    else:
        arr = pd.array([10, 11], dtype='Int64')
    if using_copy_on_write or warn_copy_on_write:
        subset['a'] = arr
    else:
        with pd.option_context('chained_assignment', 'warn'):
            with tm.assert_produces_warning(SettingWithCopyWarning):
                subset['a'] = arr
    subset._mgr._verify_integrity()
    expected = DataFrame({'a': [10, 11], 'b': [5, 6], 'c': [0.2, 0.3]}, index=range(1, 3))
    tm.assert_frame_equal(subset, expected)
    tm.assert_frame_equal(df, df_orig)