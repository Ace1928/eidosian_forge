import numpy as np
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_clip_inplace(using_copy_on_write):
    df = DataFrame({'a': [1.5, 2, 3]})
    arr_a = get_array(df, 'a')
    df.clip(lower=2, inplace=True)
    assert np.shares_memory(get_array(df, 'a'), arr_a)
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)