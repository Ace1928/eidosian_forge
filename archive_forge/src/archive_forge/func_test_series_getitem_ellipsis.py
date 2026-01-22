import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_series_getitem_ellipsis(using_copy_on_write, warn_copy_on_write):
    s = Series([1, 2, 3])
    s_orig = s.copy()
    subset = s[...]
    assert np.shares_memory(get_array(subset), get_array(s))
    with tm.assert_cow_warning(warn_copy_on_write):
        subset.iloc[0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(subset), get_array(s))
    expected = Series([0, 2, 3])
    tm.assert_series_equal(subset, expected)
    if using_copy_on_write:
        tm.assert_series_equal(s, s_orig)
    else:
        assert s.iloc[0] == 0