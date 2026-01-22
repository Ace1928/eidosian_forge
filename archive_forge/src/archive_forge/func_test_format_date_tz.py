from datetime import datetime
import dateutil.tz
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_format_date_tz(self):
    dti = pd.to_datetime([datetime(2013, 1, 1)], utc=True)
    msg = 'DatetimeIndex.format is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        formatted = dti.format()
    assert formatted[0] == '2013-01-01 00:00:00+00:00'
    dti = pd.to_datetime([datetime(2013, 1, 1), NaT], utc=True)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        formatted = dti.format()
    assert formatted[0] == '2013-01-01 00:00:00+00:00'