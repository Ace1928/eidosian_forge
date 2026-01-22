from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import delta_to_tick
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import INT_NEG_999_TO_POS_999
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries import offsets
from pandas.tseries.offsets import (
@pytest.mark.parametrize('kls, expected', [(Hour, Timedelta(hours=5)), (Minute, Timedelta(hours=2, minutes=3)), (Second, Timedelta(hours=2, seconds=3)), (Milli, Timedelta(hours=2, milliseconds=3)), (Micro, Timedelta(hours=2, microseconds=3)), (Nano, Timedelta(hours=2, nanoseconds=3))])
def test_tick_addition(kls, expected):
    offset = kls(3)
    td = Timedelta(hours=2)
    for other in [td, td.to_pytimedelta(), td.to_timedelta64()]:
        result = offset + other
        assert isinstance(result, Timedelta)
        assert result == expected
        result = other + offset
        assert isinstance(result, Timedelta)
        assert result == expected