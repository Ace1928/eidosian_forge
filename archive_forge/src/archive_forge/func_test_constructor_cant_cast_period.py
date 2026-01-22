import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_cant_cast_period(self):
    msg = 'Cannot cast PeriodIndex to dtype float64'
    with pytest.raises(TypeError, match=msg):
        Series(period_range('2000-01-01', periods=10, freq='D'), dtype=float)