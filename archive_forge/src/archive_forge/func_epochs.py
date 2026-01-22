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
@pytest.fixture(params=['timestamp', 'pydatetime', 'datetime64', 'str_1960'])
def epochs(epoch_1960, request):
    """Timestamp at 1960-01-01 in various forms.

    * Timestamp
    * datetime.datetime
    * numpy.datetime64
    * str
    """
    assert request.param in {'timestamp', 'pydatetime', 'datetime64', 'str_1960'}
    if request.param == 'timestamp':
        return epoch_1960
    elif request.param == 'pydatetime':
        return epoch_1960.to_pydatetime()
    elif request.param == 'datetime64':
        return epoch_1960.to_datetime64()
    else:
        return str(epoch_1960)