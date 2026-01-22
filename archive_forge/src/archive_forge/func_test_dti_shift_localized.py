from datetime import datetime
import pytest
import pytz
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_dti_shift_localized(self, tzstr, unit):
    dr = date_range('2011/1/1', '2012/1/1', freq='W-FRI', unit=unit)
    dr_tz = dr.tz_localize(tzstr)
    result = dr_tz.shift(1, '10min')
    assert result.tz == dr_tz.tz