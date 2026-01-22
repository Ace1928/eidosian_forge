from datetime import (
import operator
import numpy as np
import pytest
from pandas import Timestamp
import pandas._testing as tm
def test_cant_compare_tz_naive_w_aware(self, utc_fixture):
    a = Timestamp('3/12/2012')
    b = Timestamp('3/12/2012', tz=utc_fixture)
    msg = 'Cannot compare tz-naive and tz-aware timestamps'
    assert not a == b
    assert a != b
    with pytest.raises(TypeError, match=msg):
        a < b
    with pytest.raises(TypeError, match=msg):
        a <= b
    with pytest.raises(TypeError, match=msg):
        a > b
    with pytest.raises(TypeError, match=msg):
        a >= b
    assert not b == a
    assert b != a
    with pytest.raises(TypeError, match=msg):
        b < a
    with pytest.raises(TypeError, match=msg):
        b <= a
    with pytest.raises(TypeError, match=msg):
        b > a
    with pytest.raises(TypeError, match=msg):
        b >= a
    assert not a == b.to_pydatetime()
    assert not a.to_pydatetime() == b