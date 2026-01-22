from datetime import (
from hypothesis import given
import numpy as np
import pytest
from pandas._libs.tslibs import ccalendar
from pandas._testing._hypothesis import DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ
@pytest.mark.parametrize('date_tuple,expected', [((2001, 3, 1), 60), ((2004, 3, 1), 61), ((1907, 12, 31), 365), ((2004, 12, 31), 366)])
def test_get_day_of_year_numeric(date_tuple, expected):
    assert ccalendar.get_day_of_year(*date_tuple) == expected