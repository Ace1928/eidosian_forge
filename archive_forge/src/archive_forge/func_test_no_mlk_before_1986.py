from datetime import datetime
from pandas import DatetimeIndex
import pandas._testing as tm
from pandas.tseries.holiday import (
def test_no_mlk_before_1986():

    class MLKCalendar(AbstractHolidayCalendar):
        rules = [USMartinLutherKingJr]
    holidays = MLKCalendar().holidays(start='1984', end='1988').to_pydatetime().tolist()
    assert holidays == [datetime(1986, 1, 20, 0, 0), datetime(1987, 1, 19, 0, 0)]