import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_dataframe_array_string_dtype(using_copy_on_write, using_array_manager):
    df = DataFrame({'a': ['a', 'b']}, dtype='string')
    arr = np.asarray(df)
    if not using_array_manager:
        assert np.shares_memory(arr, get_array(df, 'a'))
    if using_copy_on_write:
        assert arr.flags.writeable is False
    else:
        assert arr.flags.writeable is True