import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_tz_not_aligned(self):
    ts = pd.to_datetime([1, 2]).tz_localize('UTC')
    a = DataFrame({'A': ts})
    b = DataFrame({'A': ts, 'B': ts})
    result = concat([a, b], sort=True, ignore_index=True)
    expected = DataFrame({'A': list(ts) + list(ts), 'B': [pd.NaT, pd.NaT] + list(ts)})
    tm.assert_frame_equal(result, expected)