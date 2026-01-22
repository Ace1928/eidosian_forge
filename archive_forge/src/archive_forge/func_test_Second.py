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
def test_Second():
    assert_offset_equal(Second(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 1))
    assert_offset_equal(Second(-1), datetime(2010, 1, 1, 0, 0, 1), datetime(2010, 1, 1))
    assert_offset_equal(2 * Second(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 2))
    assert_offset_equal(-1 * Second(), datetime(2010, 1, 1, 0, 0, 1), datetime(2010, 1, 1))
    assert Second(3) + Second(2) == Second(5)
    assert Second(3) - Second(2) == Second()