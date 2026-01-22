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
def test_pi_add_iadd_timedeltalike_hourly(self, two_hours):
    other = two_hours
    rng = period_range('2014-01-01 10:00', '2014-01-05 10:00', freq='h')
    expected = period_range('2014-01-01 12:00', '2014-01-05 12:00', freq='h')
    result = rng + other
    tm.assert_index_equal(result, expected)
    rng += other
    tm.assert_index_equal(rng, expected)