from datetime import (
import dateutil.tz
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas.compat import is_platform_windows
from pandas import Timestamp
@pytest.mark.parametrize('tz_name', list(pytz.common_timezones))
def test_cache_keys_are_distinct_for_pytz_vs_dateutil(tz_name):
    tz_p = timezones.maybe_get_tz(tz_name)
    tz_d = timezones.maybe_get_tz('dateutil/' + tz_name)
    if tz_d is None:
        pytest.skip(tz_name + ': dateutil does not know about this one')
    if not (tz_name == 'UTC' and is_platform_windows()):
        assert timezones._p_tz_cache_key(tz_p) != timezones._p_tz_cache_key(tz_d)