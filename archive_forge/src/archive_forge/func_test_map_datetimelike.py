from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('col, val', [['datetime', Timestamp('20130101')], ['timedelta', pd.Timedelta('1 min')]])
def test_map_datetimelike(col, val):
    df = DataFrame(np.random.default_rng(2).random((3, 4)))
    df[col] = val
    result = df.map(str)
    assert result.loc[0, col] == str(df.loc[0, col])