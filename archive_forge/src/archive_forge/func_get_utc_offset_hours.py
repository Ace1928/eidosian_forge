from datetime import timedelta
import pytest
import pytz
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas.errors import PerformanceWarning
from pandas import DatetimeIndex
import pandas._testing as tm
from pandas.util.version import Version
def get_utc_offset_hours(ts):
    o = ts.utcoffset()
    return (o.days * 24 * 3600 + o.seconds) / 3600.0