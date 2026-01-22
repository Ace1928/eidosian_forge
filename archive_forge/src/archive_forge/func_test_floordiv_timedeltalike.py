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
def test_floordiv_timedeltalike(self, td):
    assert td // td == 1
    assert 2.5 * td // td == 2
    other = Timedelta(td._value)
    msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        td // other
    res = other.to_pytimedelta() // td
    assert res == 0
    left = Timedelta._from_value_and_reso(50050, NpyDatetimeUnit.NPY_FR_us.value)
    right = Timedelta._from_value_and_reso(50, NpyDatetimeUnit.NPY_FR_ms.value)
    result = left // right
    assert result == 1
    result = right // left
    assert result == 0