from datetime import datetime
import pytest
from pytz import utc
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
@pytest.mark.parametrize('name,kwargs', [('One-Time', {'year': 2012, 'month': 5, 'day': 28}), ('Range', {'month': 5, 'day': 28, 'start_date': datetime(2012, 1, 1), 'end_date': datetime(2012, 12, 31), 'offset': DateOffset(weekday=MO(1))})])
def test_special_holidays(name, kwargs):
    base_date = [datetime(2012, 5, 28)]
    holiday = Holiday(name, **kwargs)
    start_date = datetime(2011, 1, 1)
    end_date = datetime(2020, 12, 31)
    assert base_date == holiday.dates(start_date, end_date)