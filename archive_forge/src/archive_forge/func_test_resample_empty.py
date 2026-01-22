from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_resample_empty():
    df = DataFrame(index=pd.to_datetime(['2018-01-01 00:00:00', '2018-01-01 12:00:00', '2018-01-02 00:00:00']))
    expected = DataFrame(index=pd.to_datetime(['2018-01-01 00:00:00', '2018-01-01 08:00:00', '2018-01-01 16:00:00', '2018-01-02 00:00:00']))
    result = df.resample('8h').mean()
    tm.assert_frame_equal(result, expected)