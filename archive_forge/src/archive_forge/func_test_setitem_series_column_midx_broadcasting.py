import numpy as np
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_setitem_series_column_midx_broadcasting(using_copy_on_write):
    df = DataFrame([[1, 2, 3], [3, 4, 5]], columns=MultiIndex.from_arrays([['a', 'a', 'b'], [1, 2, 3]]))
    rhs = Series([10, 11])
    df['a'] = rhs
    assert not np.shares_memory(get_array(rhs), df._get_column_array(0))
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)