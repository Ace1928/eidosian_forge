from datetime import timedelta
import sys
from hypothesis import (
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
def test_implementation_limits(self):
    min_td = Timedelta(Timedelta.min)
    max_td = Timedelta(Timedelta.max)
    assert min_td._value == iNaT + 1
    assert max_td._value == lib.i8max
    assert min_td - Timedelta(1, 'ns') is NaT
    msg = 'int too (large|big) to convert'
    with pytest.raises(OverflowError, match=msg):
        min_td - Timedelta(2, 'ns')
    with pytest.raises(OverflowError, match=msg):
        max_td + Timedelta(1, 'ns')
    td = Timedelta(min_td._value - 1, 'ns')
    assert td is NaT
    msg = "Cannot cast -9223372036854775809 from ns to 'ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        Timedelta(min_td._value - 2, 'ns')
    msg = "Cannot cast 9223372036854775808 from ns to 'ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        Timedelta(max_td._value + 1, 'ns')