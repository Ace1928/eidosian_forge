from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
def test_subtract_timestamp_from_different_timezone(self):
    t1 = Timestamp('20130101').tz_localize('US/Eastern')
    t2 = Timestamp('20130101').tz_localize('CET')
    result = t1 - t2
    assert isinstance(result, Timedelta)
    assert result == Timedelta('0 days 06:00:00')