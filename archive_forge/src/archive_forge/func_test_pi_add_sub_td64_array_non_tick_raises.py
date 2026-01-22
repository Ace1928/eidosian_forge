import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
def test_pi_add_sub_td64_array_non_tick_raises(self):
    rng = period_range('1/1/2000', freq='Q', periods=3)
    tdi = TimedeltaIndex(['-1 Day', '-1 Day', '-1 Day'])
    tdarr = tdi.values
    msg = 'Cannot add or subtract timedelta64\\[ns\\] dtype from period\\[Q-DEC\\]'
    with pytest.raises(TypeError, match=msg):
        rng + tdarr
    with pytest.raises(TypeError, match=msg):
        tdarr + rng
    with pytest.raises(TypeError, match=msg):
        rng - tdarr
    msg = 'cannot subtract PeriodArray from TimedeltaArray'
    with pytest.raises(TypeError, match=msg):
        tdarr - rng