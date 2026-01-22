import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('vals', [[1, np.nan, 2], [Timestamp('2019-12-31'), NaT, Timestamp('2020-12-31')]])
def test_interpolate_inplace_no_reference_no_copy(using_copy_on_write, vals):
    df = DataFrame({'a': vals})
    arr = get_array(df, 'a')
    df.interpolate(method='linear', inplace=True)
    assert np.shares_memory(arr, get_array(df, 'a'))
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)