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
@pytest.mark.parametrize('cls', tick_classes)
def test_tick_rdiv(cls):
    off = cls(10)
    delta = off._as_pd_timedelta
    td64 = delta.to_timedelta64()
    instance__type = '.'.join([cls.__module__, cls.__name__])
    msg = f"unsupported operand type\\(s\\) for \\/: 'int'|'float' and '{instance__type}'"
    with pytest.raises(TypeError, match=msg):
        2 / off
    with pytest.raises(TypeError, match=msg):
        2.0 / off
    assert td64 * 2.5 / off == 2.5
    if cls is not Nano:
        assert delta.to_pytimedelta() * 2 / off == 2
    result = np.array([2 * td64, td64]) / off
    expected = np.array([2.0, 1.0])
    tm.assert_numpy_array_equal(result, expected)