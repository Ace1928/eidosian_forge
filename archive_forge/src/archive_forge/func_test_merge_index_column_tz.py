import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_merge_index_column_tz(self):
    index = pd.date_range('2019-10-01', freq='30min', periods=5, tz='UTC')
    left = pd.DataFrame([0.9, 0.8, 0.7, 0.6], columns=['xyz'], index=index[1:])
    right = pd.DataFrame({'from_date': index, 'abc': [2.46] * 4 + [2.19]})
    result = merge_asof(left=left, right=right, left_index=True, right_on=['from_date'])
    expected = pd.DataFrame({'xyz': [0.9, 0.8, 0.7, 0.6], 'from_date': index[1:], 'abc': [2.46] * 3 + [2.19]}, index=pd.date_range('2019-10-01 00:30:00', freq='30min', periods=4, tz='UTC'))
    tm.assert_frame_equal(result, expected)
    result = merge_asof(left=right, right=left, right_index=True, left_on=['from_date'])
    expected = pd.DataFrame({'from_date': index, 'abc': [2.46] * 4 + [2.19], 'xyz': [np.nan, 0.9, 0.8, 0.7, 0.6]}, index=Index([0, 1, 2, 3, 4]))
    tm.assert_frame_equal(result, expected)