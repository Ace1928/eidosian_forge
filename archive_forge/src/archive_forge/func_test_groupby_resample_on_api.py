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
def test_groupby_resample_on_api():
    df = DataFrame({'key': ['A', 'B'] * 5, 'dates': date_range('2016-01-01', periods=10), 'values': np.random.default_rng(2).standard_normal(10)})
    expected = df.set_index('dates').groupby('key').resample('D').mean()
    result = df.groupby('key').resample('D', on='dates').mean()
    tm.assert_frame_equal(result, expected)