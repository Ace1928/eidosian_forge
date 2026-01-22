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
def test_tick_delta_overflow():
    tick = offsets.Day(10 ** 9)
    msg = "Cannot cast 1000000000 days 00:00:00 to unit='ns' without overflow"
    depr_msg = 'Day.delta is deprecated'
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            tick.delta