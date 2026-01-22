from datetime import datetime
import pytest
from pandas._libs.tslibs.ccalendar import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp
@pytest.mark.parametrize('n,expected', [(42, {29: 42, 1: 42, 31: 41}), (-4, {29: -4, 1: -3, 31: -4})])
@pytest.mark.parametrize('compare', [29, 1, 31])
def test_roll_convention(n, expected, compare):
    assert liboffsets.roll_convention(29, n, compare) == expected[compare]