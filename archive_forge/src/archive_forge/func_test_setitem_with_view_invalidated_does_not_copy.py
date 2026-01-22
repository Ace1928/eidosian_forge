import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_setitem_with_view_invalidated_does_not_copy(using_copy_on_write, warn_copy_on_write, request):
    df = DataFrame({'a': [1, 2, 3], 'b': 1, 'c': 1})
    view = df[:]
    df['b'] = 100
    arr = get_array(df, 'a')
    view = None
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 100
    if using_copy_on_write:
        mark = pytest.mark.xfail(reason='blk.delete does not track references correctly')
        request.applymarker(mark)
        assert np.shares_memory(arr, get_array(df, 'a'))