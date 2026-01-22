from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_after_normalize(self, unit):
    result = DatetimeIndex(date_range('2000', periods=2).as_unit(unit).normalize(), freq='D')
    expected = DatetimeIndex(['2000-01-01', '2000-01-02'], freq='D').as_unit(unit)
    tm.assert_index_equal(result, expected)