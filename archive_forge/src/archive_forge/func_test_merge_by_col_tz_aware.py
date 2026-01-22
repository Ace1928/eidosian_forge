import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_merge_by_col_tz_aware(self):
    left = pd.DataFrame({'by_col': pd.DatetimeIndex(['2018-01-01']).tz_localize('UTC'), 'on_col': [2], 'values': ['a']})
    right = pd.DataFrame({'by_col': pd.DatetimeIndex(['2018-01-01']).tz_localize('UTC'), 'on_col': [1], 'values': ['b']})
    result = merge_asof(left, right, by='by_col', on='on_col')
    expected = pd.DataFrame([[pd.Timestamp('2018-01-01', tz='UTC'), 2, 'a', 'b']], columns=['by_col', 'on_col', 'values_x', 'values_y'])
    tm.assert_frame_equal(result, expected)