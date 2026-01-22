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
def test_MillisecondTimestampArithmetic():
    assert_offset_equal(Milli(), Timestamp('2010-01-01'), Timestamp('2010-01-01 00:00:00.001'))
    assert_offset_equal(Milli(-1), Timestamp('2010-01-01 00:00:00.001'), Timestamp('2010-01-01'))