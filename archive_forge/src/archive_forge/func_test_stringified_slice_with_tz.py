from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_stringified_slice_with_tz(self):
    start = '2013-01-07'
    idx = date_range(start=start, freq='1d', periods=10, tz='US/Eastern')
    df = DataFrame(np.arange(10), index=idx)
    df['2013-01-14 23:44:34.437768-05:00':]