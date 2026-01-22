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
def test_delta_to_tick():
    delta = timedelta(3)
    tick = delta_to_tick(delta)
    assert tick == offsets.Day(3)
    td = Timedelta(nanoseconds=5)
    tick = delta_to_tick(td)
    assert tick == Nano(5)