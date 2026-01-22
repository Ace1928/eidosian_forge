from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
def test_overflow_on_construction():
    value = Timedelta('1day')._value * 20169940
    msg = "Cannot cast 1742682816000000000000 from ns to 'ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        Timedelta(value)
    msg = "Cannot cast 139993 from D to 'ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        Timedelta(7 * 19999, unit='D')
    td = Timedelta(timedelta(days=13 * 19999))
    assert td._creso == NpyDatetimeUnit.NPY_FR_us.value
    assert td.days == 13 * 19999