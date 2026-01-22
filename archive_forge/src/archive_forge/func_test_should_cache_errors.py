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
@pytest.mark.parametrize('unique_share,check_count, err_message', [(0.5, 11, 'check_count must be in next bounds: \\[0; len\\(arg\\)\\]'), (10, 2, 'unique_share must be in next bounds: \\(0; 1\\)')])
def test_should_cache_errors(self, unique_share, check_count, err_message):
    arg = [5] * 10
    with pytest.raises(AssertionError, match=err_message):
        tools.should_cache(arg, unique_share, check_count)