import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('key, level', [('l1', 0), (2, 1)])
def test_xs_multiindex(using_copy_on_write, warn_copy_on_write, using_array_manager, key, level, axis):
    arr = np.arange(18).reshape(6, 3)
    index = MultiIndex.from_product([['l1', 'l2'], [1, 2, 3]], names=['lev1', 'lev2'])
    df = DataFrame(arr, index=index, columns=list('abc'))
    if axis == 1:
        df = df.transpose().copy()
    df_orig = df.copy()
    result = df.xs(key, level=level, axis=axis)
    if level == 0:
        assert np.shares_memory(get_array(df, df.columns[0]), get_array(result, result.columns[0]))
    if warn_copy_on_write:
        warn = FutureWarning if level == 0 else None
    elif not using_copy_on_write and (not using_array_manager):
        warn = SettingWithCopyWarning
    else:
        warn = None
    with option_context('chained_assignment', 'warn'):
        with tm.assert_produces_warning(warn):
            result.iloc[0, 0] = 0
    tm.assert_frame_equal(df, df_orig)