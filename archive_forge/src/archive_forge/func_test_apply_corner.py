from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
def test_apply_corner(self, _offset):
    if _offset is BDay:
        msg = 'Only know how to combine business day with datetime or timedelta'
    else:
        msg = 'Only know how to combine trading day with datetime, datetime64 or timedelta'
    with pytest.raises(ApplyTypeError, match=msg):
        _offset()._apply(BMonthEnd())