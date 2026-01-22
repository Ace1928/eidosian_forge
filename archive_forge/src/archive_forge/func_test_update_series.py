import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_update_series(using_copy_on_write, warn_copy_on_write):
    ser1 = Series([1.0, 2.0, 3.0])
    ser2 = Series([100.0], index=[1])
    ser1_orig = ser1.copy()
    view = ser1[:]
    if warn_copy_on_write:
        with tm.assert_cow_warning():
            ser1.update(ser2)
    else:
        ser1.update(ser2)
    expected = Series([1.0, 100.0, 3.0])
    tm.assert_series_equal(ser1, expected)
    if using_copy_on_write:
        tm.assert_series_equal(view, ser1_orig)
    else:
        tm.assert_series_equal(view, expected)