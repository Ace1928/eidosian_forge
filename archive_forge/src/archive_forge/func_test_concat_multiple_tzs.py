import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_multiple_tzs(self):
    ts1 = Timestamp('2015-01-01', tz=None)
    ts2 = Timestamp('2015-01-01', tz='UTC')
    ts3 = Timestamp('2015-01-01', tz='EST')
    df1 = DataFrame({'time': [ts1]})
    df2 = DataFrame({'time': [ts2]})
    df3 = DataFrame({'time': [ts3]})
    results = concat([df1, df2]).reset_index(drop=True)
    expected = DataFrame({'time': [ts1, ts2]}, dtype=object)
    tm.assert_frame_equal(results, expected)
    results = concat([df1, df3]).reset_index(drop=True)
    expected = DataFrame({'time': [ts1, ts3]}, dtype=object)
    tm.assert_frame_equal(results, expected)
    results = concat([df2, df3]).reset_index(drop=True)
    expected = DataFrame({'time': [ts2, ts3]})
    tm.assert_frame_equal(results, expected)