import pytest
from pandas import (
@pytest.mark.parametrize('tz_left, tz_right', [(None, 'UTC'), ('UTC', None), ('UTC', 'US/Eastern')])
def test_constructor_errors_tz(self, tz_left, tz_right):
    left = Timestamp('2017-01-01', tz=tz_left)
    right = Timestamp('2017-01-02', tz=tz_right)
    if tz_left is None or tz_right is None:
        error = TypeError
        msg = 'Cannot compare tz-naive and tz-aware timestamps'
    else:
        error = ValueError
        msg = 'left and right must have the same time zone'
    with pytest.raises(error, match=msg):
        Interval(left, right)