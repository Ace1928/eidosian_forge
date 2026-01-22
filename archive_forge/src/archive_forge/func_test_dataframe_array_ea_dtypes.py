import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_dataframe_array_ea_dtypes(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3]}, dtype='Int64')
    arr = np.asarray(df, dtype='int64')
    assert np.shares_memory(arr, get_array(df, 'a'))
    if using_copy_on_write:
        assert arr.flags.writeable is False
    else:
        assert arr.flags.writeable is True
    arr = np.asarray(df)
    assert np.shares_memory(arr, get_array(df, 'a'))
    if using_copy_on_write:
        assert arr.flags.writeable is False
    else:
        assert arr.flags.writeable is True