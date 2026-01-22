from datetime import datetime
import pytest
from pytz import utc
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
@pytest.mark.parametrize('transform', [lambda x: x.strftime('%Y-%m-%d'), lambda x: Timestamp(x)])
def test_argument_types(transform):
    start_date = datetime(2011, 1, 1)
    end_date = datetime(2020, 12, 31)
    holidays = USThanksgivingDay.dates(start_date, end_date)
    holidays2 = USThanksgivingDay.dates(transform(start_date), transform(end_date))
    tm.assert_index_equal(holidays, holidays2)