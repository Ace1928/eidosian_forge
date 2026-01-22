from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_groupby_first_datetime64(self):
    df = DataFrame([(1, 1351036800000000000), (2, 1351036800000000000)])
    df[1] = df[1].astype('M8[ns]')
    assert issubclass(df[1].dtype.type, np.datetime64)
    result = df.groupby(level=0).first()
    got_dt = result[1].dtype
    assert issubclass(got_dt.type, np.datetime64)
    result = df[1].groupby(level=0).first()
    got_dt = result.dtype
    assert issubclass(got_dt.type, np.datetime64)