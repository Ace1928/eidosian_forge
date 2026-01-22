from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
@pytest.fixture
def offset8():
    return BusinessHour(start=['09:00', '13:00'], end=['12:00', '17:00'])