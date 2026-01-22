from datetime import datetime
import pytest
from pandas.tseries.holiday import (
@pytest.mark.parametrize('day,expected', [(_SATURDAY, _MONDAY), (_SUNDAY, _TUESDAY), (_MONDAY, _TUESDAY)])
def test_next_monday_or_tuesday(day, expected):
    assert next_monday_or_tuesday(day) == expected