import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_droplevel_false(self):
    df = DataFrame([[1, 2, 3]], columns=Index(['a', 'b', 'c']))
    result = df.xs('a', axis=1, drop_level=False)
    expected = DataFrame({'a': [1]})
    tm.assert_frame_equal(result, expected)