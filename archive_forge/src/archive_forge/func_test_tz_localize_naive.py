from datetime import timezone
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_tz_localize_naive(self, frame_or_series):
    rng = date_range('1/1/2011', periods=100, freq='h', tz='utc')
    ts = Series(1, index=rng)
    ts = frame_or_series(ts)
    with pytest.raises(TypeError, match='Already tz-aware'):
        ts.tz_localize('US/Eastern')