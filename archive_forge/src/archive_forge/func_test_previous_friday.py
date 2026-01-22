from datetime import datetime
import pytest
from pandas.tseries.holiday import (
@pytest.mark.parametrize('day', [_SATURDAY, _SUNDAY])
def test_previous_friday(day):
    assert previous_friday(day) == _FRIDAY