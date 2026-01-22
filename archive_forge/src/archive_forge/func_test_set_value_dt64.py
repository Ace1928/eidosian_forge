from datetime import datetime
import numpy as np
from pandas import (
import pandas._testing as tm
def test_set_value_dt64(datetime_series):
    idx = datetime_series.index[10]
    res = datetime_series._set_value(idx, 0)
    assert res is None
    assert datetime_series[idx] == 0