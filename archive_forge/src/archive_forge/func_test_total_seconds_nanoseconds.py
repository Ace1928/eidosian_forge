from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
def test_total_seconds_nanoseconds(self):
    start_time = pd.Series(['2145-11-02 06:00:00']).astype('datetime64[ns]')
    end_time = pd.Series(['2145-11-02 07:06:00']).astype('datetime64[ns]')
    expected = (end_time - start_time).values / np.timedelta64(1, 's')
    result = (end_time - start_time).dt.total_seconds().values
    assert result == expected