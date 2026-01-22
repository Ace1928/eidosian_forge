import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@td.skip_array_manager_invalid_test
@pytest.mark.parametrize('dtype', [np.intp, np.int8])
@pytest.mark.parametrize('locs, arr', [([0], np.array([-1, -2, -3])), ([1], np.array([-1, -2, -3])), ([5], np.array([-1, -2, -3])), ([0, 1], np.array([[-1, -2, -3], [-4, -5, -6]]).T), ([0, 2], np.array([[-1, -2, -3], [-4, -5, -6]]).T), ([0, 1, 2], np.array([[-1, -2, -3], [-4, -5, -6], [-4, -5, -6]]).T), ([1, 2], np.array([[-1, -2, -3], [-4, -5, -6]]).T), ([1, 3], np.array([[-1, -2, -3], [-4, -5, -6]]).T), ([1, 3], np.array([[-1, -2, -3], [-4, -5, -6]]).T)])
def test_iset_splits_blocks_inplace(using_copy_on_write, locs, arr, dtype):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'd': [10, 11, 12], 'e': [13, 14, 15], 'f': ['a', 'b', 'c']})
    arr = arr.astype(dtype)
    df_orig = df.copy()
    df2 = df.copy(deep=None)
    df2._mgr.iset(locs, arr, inplace=True)
    tm.assert_frame_equal(df, df_orig)
    if using_copy_on_write:
        for i, col in enumerate(df.columns):
            if i not in locs:
                assert np.shares_memory(get_array(df, col), get_array(df2, col))
    else:
        for col in df.columns:
            assert not np.shares_memory(get_array(df, col), get_array(df2, col))