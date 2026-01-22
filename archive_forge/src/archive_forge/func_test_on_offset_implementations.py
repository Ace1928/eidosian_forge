from hypothesis import (
import pytest
import pytz
import pandas as pd
from pandas._testing._hypothesis import (
@pytest.mark.arm_slow
@given(DATETIME_JAN_1_1900_OPTIONAL_TZ, YQM_OFFSET)
def test_on_offset_implementations(dt, offset):
    assume(not offset.normalize)
    try:
        compare = dt + offset - offset
    except (pytz.NonExistentTimeError, pytz.AmbiguousTimeError):
        assume(False)
    assert offset.is_on_offset(dt) == (compare == dt)