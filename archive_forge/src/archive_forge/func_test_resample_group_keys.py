from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_resample_group_keys():
    df = DataFrame({'A': 1, 'B': 2}, index=date_range('2000', periods=10))
    expected = df.copy()
    g = df.resample('5D', group_keys=False)
    result = g.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)
    g = df.resample('5D')
    result = g.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)
    expected.index = pd.MultiIndex.from_arrays([pd.to_datetime(['2000-01-01', '2000-01-06']).as_unit('ns').repeat(5), expected.index])
    g = df.resample('5D', group_keys=True)
    result = g.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)