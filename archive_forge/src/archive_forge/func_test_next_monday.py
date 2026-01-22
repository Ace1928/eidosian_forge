from datetime import datetime
import pytest
from pandas.tseries.holiday import (
@pytest.mark.parametrize('day', [_SATURDAY, _SUNDAY])
def test_next_monday(day):
    assert next_monday(day) == _MONDAY