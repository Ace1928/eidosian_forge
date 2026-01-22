from datetime import datetime
import dateutil.tz
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_format_with_name_time_info(self):
    dates = pd.date_range('2011-01-01 04:00:00', periods=10, name='something')
    msg = 'DatetimeIndex.format is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        formatted = dates.format(name=True)
    assert formatted[0] == 'something'