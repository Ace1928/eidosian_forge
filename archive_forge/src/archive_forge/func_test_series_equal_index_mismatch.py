import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('check_index', [True, False])
def test_series_equal_index_mismatch(check_index):
    s1 = Series([1, 2, 3], index=['a', 'b', 'c'])
    s2 = Series([1, 2, 3], index=['c', 'b', 'a'])
    if check_index:
        with pytest.raises(AssertionError, match='Series.index are different'):
            tm.assert_series_equal(s1, s2, check_index=check_index)
    else:
        _assert_series_equal_both(s1, s2, check_index=check_index)