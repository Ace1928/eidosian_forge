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
def test_to_timedelta64(self, td, unit):
    for res in [td.to_timedelta64(), td.to_numpy(), td.asm8]:
        assert isinstance(res, np.timedelta64)
        assert res.view('i8') == td._value
        if unit == NpyDatetimeUnit.NPY_FR_s.value:
            assert res.dtype == 'm8[s]'
        elif unit == NpyDatetimeUnit.NPY_FR_ms.value:
            assert res.dtype == 'm8[ms]'
        elif unit == NpyDatetimeUnit.NPY_FR_us.value:
            assert res.dtype == 'm8[us]'