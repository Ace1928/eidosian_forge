from datetime import datetime
import warnings
import dateutil
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
@pytest.mark.parametrize('offset', [offsets.MonthBegin(), offsets.BYearBegin(2), offsets.BusinessHour(2)])
def test_asfreq_invalid_period_offset(self, offset, series_and_frame):
    msg = f"Invalid offset: '{offset.base}' for converting time series "
    df = series_and_frame
    with pytest.raises(ValueError, match=msg):
        df.asfreq(freq=offset)