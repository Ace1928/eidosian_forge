import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_basic_categorical(self, trades, asof, quotes):
    expected = asof
    trades.ticker = trades.ticker.astype('category')
    quotes.ticker = quotes.ticker.astype('category')
    expected.ticker = expected.ticker.astype('category')
    result = merge_asof(trades, quotes, on='time', by='ticker')
    tm.assert_frame_equal(result, expected)