from datetime import (
import dateutil.tz
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas.compat import is_platform_windows
from pandas import Timestamp
def test_tzlocal_maybe_get_tz():
    tz = timezones.maybe_get_tz('tzlocal()')
    assert tz == dateutil.tz.tzlocal()