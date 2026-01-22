import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq, base_date', [('W-THU', '1970-01-01'), ('D', '1970-01-01'), ('B', '1970-01-01'), ('h', '1970-01-01'), ('min', '1970-01-01'), ('s', '1970-01-01'), ('ms', '1970-01-01'), ('us', '1970-01-01'), ('ns', '1970-01-01'), ('M', '1970-01'), ('Y', 1970)])
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
@pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
def test_freq(self, freq, base_date):
    rng = period_range(start=base_date, periods=10, freq=freq)
    exp = np.arange(10, dtype=np.int64)
    tm.assert_numpy_array_equal(rng.asi8, exp)