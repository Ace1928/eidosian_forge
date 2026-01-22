from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries.holiday import USFederalHolidayCalendar
@pytest.mark.parametrize('nano_case', nano_cases)
def test_apply_nanoseconds(self, nano_case):
    offset, cases = nano_case
    for base, expected in cases.items():
        assert_offset_equal(offset, base, expected)