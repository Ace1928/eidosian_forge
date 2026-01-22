from datetime import (
import dateutil.tz
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas.compat import is_platform_windows
from pandas import Timestamp
def test_infer_tz_utc_localize(infer_setup):
    _, _, start, end, start_naive, end_naive = infer_setup
    utc = pytz.utc
    start = utc.localize(start_naive)
    end = utc.localize(end_naive)
    assert timezones.infer_tzinfo(start, end) is utc