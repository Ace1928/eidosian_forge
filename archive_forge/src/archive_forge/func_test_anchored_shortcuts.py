import re
import pytest
from pandas._libs.tslibs import (
@pytest.mark.parametrize('shortcut,expected', [('W', offsets.Week(weekday=6)), ('W-SUN', offsets.Week(weekday=6)), ('QE', offsets.QuarterEnd(startingMonth=12)), ('QE-DEC', offsets.QuarterEnd(startingMonth=12)), ('QE-MAY', offsets.QuarterEnd(startingMonth=5)), ('SME', offsets.SemiMonthEnd(day_of_month=15)), ('SME-15', offsets.SemiMonthEnd(day_of_month=15)), ('SME-1', offsets.SemiMonthEnd(day_of_month=1)), ('SME-27', offsets.SemiMonthEnd(day_of_month=27)), ('SMS-2', offsets.SemiMonthBegin(day_of_month=2)), ('SMS-27', offsets.SemiMonthBegin(day_of_month=27))])
def test_anchored_shortcuts(shortcut, expected):
    result = to_offset(shortcut)
    assert result == expected