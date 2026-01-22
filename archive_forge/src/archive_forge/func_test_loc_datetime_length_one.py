from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_loc_datetime_length_one(self):
    df = DataFrame(columns=['1'], index=date_range('2016-10-01T00:00:00', '2016-10-01T23:59:59'))
    result = df.loc[datetime(2016, 10, 1):]
    tm.assert_frame_equal(result, df)
    result = df.loc['2016-10-01T00:00:00':]
    tm.assert_frame_equal(result, df)