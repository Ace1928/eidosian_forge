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
def test_pi_offset_errors(self):
    idx = PeriodIndex(['2011-01-01', '2011-02-01', '2011-03-01', '2011-04-01'], freq='D', name='idx')
    ser = Series(idx)
    msg = "Cannot add/subtract timedelta-like from PeriodArray that is not an integer multiple of the PeriodArray's freq"
    for obj in [idx, ser]:
        with pytest.raises(IncompatibleFrequency, match=msg):
            obj + pd.offsets.Hour(2)
        with pytest.raises(IncompatibleFrequency, match=msg):
            pd.offsets.Hour(2) + obj
        with pytest.raises(IncompatibleFrequency, match=msg):
            obj - pd.offsets.Hour(2)