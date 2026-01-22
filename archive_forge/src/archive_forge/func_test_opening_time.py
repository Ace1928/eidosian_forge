from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
@pytest.mark.parametrize('case', opening_time_cases)
def test_opening_time(self, case):
    _offsets, cases = case
    for offset in _offsets:
        for dt, (exp_next, exp_prev) in cases.items():
            assert offset._next_opening_time(dt) == exp_next
            assert offset._prev_opening_time(dt) == exp_prev