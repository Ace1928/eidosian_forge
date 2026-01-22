import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_del_series(backend):
    _, _, Series = backend
    s = Series([1, 2, 3], index=['a', 'b', 'c'])
    s_orig = s.copy()
    s2 = s[:]
    assert np.shares_memory(get_array(s), get_array(s2))
    del s2['a']
    assert not np.shares_memory(get_array(s), get_array(s2))
    tm.assert_series_equal(s, s_orig)
    tm.assert_series_equal(s2, s_orig[['b', 'c']])
    values = s2.values
    s2.loc['b'] = 100
    assert values[0] == 100