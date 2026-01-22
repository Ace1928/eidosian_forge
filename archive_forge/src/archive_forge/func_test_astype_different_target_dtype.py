import pickle
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', ['float64', 'int32', 'Int32', 'int32[pyarrow]'])
def test_astype_different_target_dtype(using_copy_on_write, dtype):
    if dtype == 'int32[pyarrow]':
        pytest.importorskip('pyarrow')
    df = DataFrame({'a': [1, 2, 3]})
    df_orig = df.copy()
    df2 = df.astype(dtype)
    assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    if using_copy_on_write:
        assert df2._mgr._has_no_reference(0)
    df2.iloc[0, 0] = 5
    tm.assert_frame_equal(df, df_orig)
    df2 = df.astype(dtype)
    df.iloc[0, 0] = 100
    tm.assert_frame_equal(df2, df_orig.astype(dtype))