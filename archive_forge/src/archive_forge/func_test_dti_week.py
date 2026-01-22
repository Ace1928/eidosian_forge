import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_dti_week(self):
    dates = ['2013/12/29', '2013/12/30', '2013/12/31']
    dates = DatetimeIndex(dates, tz='Europe/Brussels')
    expected = [52, 1, 1]
    assert dates.isocalendar().week.tolist() == expected
    assert [d.weekofyear for d in dates] == expected