import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_str_slice():
    df = DataFrame([['20160525 13:30:00.023', 'MSFT', '51.95', '51.95'], ['20160525 13:30:00.048', 'GOOG', '720.50', '720.93'], ['20160525 13:30:00.076', 'AAPL', '98.55', '98.56'], ['20160525 13:30:00.131', 'AAPL', '98.61', '98.62'], ['20160525 13:30:00.135', 'MSFT', '51.92', '51.95'], ['20160525 13:30:00.135', 'AAPL', '98.61', '98.62']], columns='time,ticker,bid,ask'.split(','))
    df2 = df.set_index(['ticker', 'time']).sort_index()
    res = df2.loc[('AAPL', slice('2016-05-25 13:30:00')), :].droplevel(0)
    expected = df2.loc['AAPL'].loc[slice('2016-05-25 13:30:00'), :]
    tm.assert_frame_equal(res, expected)