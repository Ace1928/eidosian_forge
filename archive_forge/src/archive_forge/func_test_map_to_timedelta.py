from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_to_timedelta():
    list_of_valid_strings = ['00:00:01', '00:00:02']
    a = pd.to_timedelta(list_of_valid_strings)
    b = Series(list_of_valid_strings).map(pd.to_timedelta)
    tm.assert_series_equal(Series(a), b)
    list_of_strings = ['00:00:01', np.nan, pd.NaT, pd.NaT]
    a = pd.to_timedelta(list_of_strings)
    ser = Series(list_of_strings)
    b = ser.map(pd.to_timedelta)
    tm.assert_series_equal(Series(a), b)