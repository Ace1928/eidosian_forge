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
def test_Microsecond():
    assert_offset_equal(Micro(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 1))
    assert_offset_equal(Micro(-1), datetime(2010, 1, 1, 0, 0, 0, 1), datetime(2010, 1, 1))
    assert_offset_equal(2 * Micro(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 2))
    assert_offset_equal(-1 * Micro(), datetime(2010, 1, 1, 0, 0, 0, 1), datetime(2010, 1, 1))
    assert Micro(3) + Micro(2) == Micro(5)
    assert Micro(3) - Micro(2) == Micro()