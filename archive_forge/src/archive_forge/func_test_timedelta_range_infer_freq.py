import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_timedelta_range_infer_freq(self):
    result = timedelta_range('0s', '1s', periods=31)
    assert result.freq is None