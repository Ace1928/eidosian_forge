from datetime import datetime
import dateutil.tz
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_format_datetime(self):
    msg = 'DatetimeIndex.format is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        formatted = pd.to_datetime([datetime(2003, 1, 1, 12), NaT]).format()
    assert formatted[0] == '2003-01-01 12:00:00'
    assert formatted[1] == 'NaT'