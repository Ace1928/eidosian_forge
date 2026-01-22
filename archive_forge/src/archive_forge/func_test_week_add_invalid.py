from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
def test_week_add_invalid(self):
    offset = Week(weekday=1)
    other = Day()
    with pytest.raises(TypeError, match='Cannot add'):
        offset + other