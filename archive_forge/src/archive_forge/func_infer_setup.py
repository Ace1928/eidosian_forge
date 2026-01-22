from datetime import (
import dateutil.tz
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas.compat import is_platform_windows
from pandas import Timestamp
@pytest.fixture(params=[(pytz.timezone('US/Eastern'), lambda tz, x: tz.localize(x)), (dateutil.tz.gettz('US/Eastern'), lambda tz, x: x.replace(tzinfo=tz))])
def infer_setup(request):
    eastern, localize = request.param
    start_naive = datetime(2001, 1, 1)
    end_naive = datetime(2009, 1, 1)
    start = localize(eastern, start_naive)
    end = localize(eastern, end_naive)
    return (eastern, localize, start, end, start_naive, end_naive)