from datetime import datetime
from dateutil.tz import gettz
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
import pandas.util._test_decorators as td
import pandas._testing as tm
@td.skip_if_windows
def test_replace_tzinfo(self):
    dt = datetime(2016, 3, 27, 1)
    tzinfo = pytz.timezone('CET').localize(dt, is_dst=False).tzinfo
    result_dt = dt.replace(tzinfo=tzinfo)
    result_pd = Timestamp(dt).replace(tzinfo=tzinfo)
    with tm.set_timezone('UTC'):
        assert result_dt.timestamp() == result_pd.timestamp()
    assert result_dt == result_pd
    assert result_dt == result_pd.to_pydatetime()
    result_dt = dt.replace(tzinfo=tzinfo).replace(tzinfo=None)
    result_pd = Timestamp(dt).replace(tzinfo=tzinfo).replace(tzinfo=None)
    with tm.set_timezone('UTC'):
        assert result_dt.timestamp() == result_pd.timestamp()
    assert result_dt == result_pd
    assert result_dt == result_pd.to_pydatetime()