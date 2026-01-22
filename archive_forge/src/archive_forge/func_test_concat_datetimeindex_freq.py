import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_datetimeindex_freq(self):
    dr = date_range('01-Jan-2013', periods=100, freq='50ms', tz='UTC')
    data = list(range(100))
    expected = DataFrame(data, index=dr)
    result = concat([expected[:50], expected[50:]])
    tm.assert_frame_equal(result, expected)
    result = concat([expected[50:], expected[:50]])
    expected = DataFrame(data[50:] + data[:50], index=dr[50:].append(dr[:50]))
    expected.index._data.freq = None
    tm.assert_frame_equal(result, expected)