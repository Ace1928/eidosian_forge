import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', ['int64', 'float64'], ids=['single-block', 'mixed-block'])
@pytest.mark.parametrize('row_indexer', [slice(1, 3), np.array([False, True, True]), np.array([1, 2])], ids=['slice', 'mask', 'array'])
@pytest.mark.parametrize('column_indexer', [slice(1, 3), np.array([False, True, True]), [1, 2]], ids=['slice', 'mask', 'array'])
def test_subset_iloc_rows_columns(backend, dtype, row_indexer, column_indexer, using_array_manager, using_copy_on_write, warn_copy_on_write):
    dtype_backend, DataFrame, _ = backend
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': np.array([7, 8, 9], dtype=dtype)})
    df_orig = df.copy()
    subset = df.iloc[row_indexer, column_indexer]
    mutate_parent = isinstance(row_indexer, slice) and isinstance(column_indexer, slice) and (using_array_manager or (dtype == 'int64' and dtype_backend == 'numpy' and (not using_copy_on_write)))
    with tm.assert_cow_warning(warn_copy_on_write and mutate_parent):
        subset.iloc[0, 0] = 0
    expected = DataFrame({'b': [0, 6], 'c': np.array([8, 9], dtype=dtype)}, index=range(1, 3))
    tm.assert_frame_equal(subset, expected)
    if mutate_parent:
        df_orig.iloc[1, 1] = 0
    tm.assert_frame_equal(df, df_orig)