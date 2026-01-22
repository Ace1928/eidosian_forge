from __future__ import annotations
from datetime import datetime
import pytest
import pandas as pd
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
def test_offsets_compare_equal(self):
    offset1 = BMonthEnd()
    offset2 = BMonthEnd()
    assert not offset1 != offset2