import pickle
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', ['int64', 'Int64'])
@pytest.mark.parametrize('new_dtype', ['int64', 'Int64', 'int64[pyarrow]'])
def test_astype_avoids_copy(using_copy_on_write, dtype, new_dtype):
    if new_dtype == 'int64[pyarrow]':
        pytest.importorskip('pyarrow')
    df = DataFrame({'a': [1, 2, 3]}, dtype=dtype)
    df_orig = df.copy()
    df2 = df.astype(new_dtype)
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    else:
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    df2.iloc[0, 0] = 10
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    tm.assert_frame_equal(df, df_orig)
    df2 = df.astype(new_dtype)
    df.iloc[0, 0] = 100
    tm.assert_frame_equal(df2, df_orig.astype(new_dtype))