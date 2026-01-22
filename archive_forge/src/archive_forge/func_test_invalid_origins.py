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
@pytest.mark.parametrize('origin, exc', [('random_string', ValueError), ('epoch', ValueError), ('13-24-1990', ValueError), (datetime(1, 1, 1), OutOfBoundsDatetime)])
def test_invalid_origins(self, origin, exc, units, units_from_epochs):
    msg = '|'.join([f'origin {origin} is Out of Bounds', f'origin {origin} cannot be converted to a Timestamp', "Cannot cast .* to unit='ns' without overflow"])
    with pytest.raises(exc, match=msg):
        to_datetime(units_from_epochs, unit=units, origin=origin)