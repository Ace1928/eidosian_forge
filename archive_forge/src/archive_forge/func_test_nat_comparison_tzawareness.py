from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_nat_comparison_tzawareness(self, comparison_op):
    op = comparison_op
    dti = DatetimeIndex(['2014-01-01', NaT, '2014-03-01', NaT, '2014-05-01', '2014-07-01'])
    expected = np.array([op == operator.ne] * len(dti))
    result = op(dti, NaT)
    tm.assert_numpy_array_equal(result, expected)
    result = op(dti.tz_localize('US/Pacific'), NaT)
    tm.assert_numpy_array_equal(result, expected)