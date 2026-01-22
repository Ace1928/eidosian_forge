from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
def test_period_comparison_nat(self):
    per = Period('2011-01-01', freq='D')
    ts = Timestamp('2011-01-01')
    for left, right in [(NaT, per), (per, NaT), (NaT, ts), (ts, NaT)]:
        assert not left < right
        assert not left > right
        assert not left == right
        assert left != right
        assert not left <= right
        assert not left >= right