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
@pytest.mark.parametrize('unit, value, expected', [('us', 9.999, 9999), ('ms', 9.999999, 9999999), ('s', 9.999999999, 9999999999)])
def test_rounding_on_int_unit_construction(self, unit, value, expected):
    result = Timedelta(value, unit=unit)
    assert result._value == expected
    result = Timedelta(str(value) + unit)
    assert result._value == expected