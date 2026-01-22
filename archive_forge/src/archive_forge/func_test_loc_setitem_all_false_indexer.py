from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_loc_setitem_all_false_indexer():
    ser = Series([1, 2], index=['a', 'b'])
    expected = ser.copy()
    rhs = Series([6, 7], index=['a', 'b'])
    ser.loc[ser > 100] = rhs
    tm.assert_series_equal(ser, expected)