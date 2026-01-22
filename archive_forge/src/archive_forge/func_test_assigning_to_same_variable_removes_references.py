import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_assigning_to_same_variable_removes_references(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3]})
    df = df.reset_index()
    if using_copy_on_write:
        assert df._mgr._has_no_reference(1)
    arr = get_array(df, 'a')
    df.iloc[0, 1] = 100
    assert np.shares_memory(arr, get_array(df, 'a'))