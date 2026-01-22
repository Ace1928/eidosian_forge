import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_with_without_freq(self):
    start = Period('2002-01-01 00:00', freq='30min')
    exp = period_range(start=start, periods=5, freq=start.freq)
    result = period_range(start=start, periods=5)
    tm.assert_index_equal(exp, result)