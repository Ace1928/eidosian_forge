from datetime import datetime
import pytest
from pandas.tseries.holiday import (
@pytest.mark.parametrize('day,expected', [(_SATURDAY, _MONDAY), (_SUNDAY, _MONDAY), (_MONDAY, _MONDAY)])
def test_weekend_to_monday(day, expected):
    assert weekend_to_monday(day) == expected