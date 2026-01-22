from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_timestamp_bug_NaT():
    frame = DataFrame([[pd.NaT, pd.NaT]], columns=['a', 'b'])
    other = DataFrame([[datetime(2020, 1, 1), datetime(2020, 1, 2)]], columns=['b', 'c'])
    result = frame.combine_first(other)
    expected = DataFrame([[pd.NaT, datetime(2020, 1, 1), datetime(2020, 1, 2)]], columns=['a', 'b', 'c'])
    tm.assert_frame_equal(result, expected)