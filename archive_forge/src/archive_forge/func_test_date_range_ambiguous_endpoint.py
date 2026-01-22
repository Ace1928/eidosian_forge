from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
@pytest.mark.parametrize('tz', ['Europe/London', 'dateutil/Europe/London'])
def test_date_range_ambiguous_endpoint(self, tz):
    with pytest.raises(pytz.AmbiguousTimeError, match='Cannot infer dst time'):
        date_range('2013-10-26 23:00', '2013-10-27 01:00', tz='Europe/London', freq='h')
    times = date_range('2013-10-26 23:00', '2013-10-27 01:00', freq='h', tz=tz, ambiguous='infer')
    assert times[0] == Timestamp('2013-10-26 23:00', tz=tz)
    assert times[-1] == Timestamp('2013-10-27 01:00:00+0000', tz=tz)