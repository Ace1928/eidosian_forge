from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
def test_from_pytimedelta_us_reso():
    td = timedelta(days=4, minutes=3)
    result = Timedelta(td)
    assert result.to_pytimedelta() == td
    assert result._creso == NpyDatetimeUnit.NPY_FR_us.value