import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewm_with_nat_raises(halflife_with_times):
    ser = Series(range(1))
    times = DatetimeIndex(['NaT'])
    with pytest.raises(ValueError, match='Cannot convert NaT values to integer'):
        ser.ewm(com=0.1, halflife=halflife_with_times, times=times)