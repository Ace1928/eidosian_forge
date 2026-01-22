import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
@pytest.mark.xfail(reason='fails to enforce dayfirst=True, which would raise')
def test_to_datetime_respects_dayfirst(self, cache):
    msg = 'Invalid date specified'
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(UserWarning, match='Provide format'):
            to_datetime('01-13-2012', dayfirst=True, cache=cache)