import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_multiindex_droplevel_false(self):
    mi = MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'x')], names=['level1', 'level2'])
    df = DataFrame([[1, 2, 3]], columns=mi)
    result = df.xs('a', axis=1, drop_level=False)
    expected = DataFrame([[1, 2]], columns=MultiIndex.from_tuples([('a', 'x'), ('a', 'y')], names=['level1', 'level2']))
    tm.assert_frame_equal(result, expected)