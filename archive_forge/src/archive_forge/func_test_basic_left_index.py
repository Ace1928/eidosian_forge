import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_basic_left_index(self, trades, asof, quotes):
    expected = asof
    trades = trades.set_index('time')
    result = merge_asof(trades, quotes, left_index=True, right_on='time', by='ticker')
    expected.index = result.index
    expected = expected[result.columns]
    tm.assert_frame_equal(result, expected)