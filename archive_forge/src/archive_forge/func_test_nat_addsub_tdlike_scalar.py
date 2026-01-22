from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
@pytest.mark.parametrize('obj', [offsets.YearEnd(2), offsets.YearBegin(2), offsets.MonthBegin(1), offsets.MonthEnd(2), offsets.MonthEnd(12), offsets.Day(2), offsets.Day(5), offsets.Hour(24), offsets.Hour(3), offsets.Minute(), np.timedelta64(3, 'h'), np.timedelta64(4, 'h'), np.timedelta64(3200, 's'), np.timedelta64(3600, 's'), np.timedelta64(3600 * 24, 's'), np.timedelta64(2, 'D'), np.timedelta64(365, 'D'), timedelta(-2), timedelta(365), timedelta(minutes=120), timedelta(days=4, minutes=180), timedelta(hours=23), timedelta(hours=23, minutes=30), timedelta(hours=48)])
def test_nat_addsub_tdlike_scalar(obj):
    assert NaT + obj is NaT
    assert obj + NaT is NaT
    assert NaT - obj is NaT