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
def test_pi_sub_pdnat(self):
    idx = PeriodIndex(['2011-01', '2011-02', 'NaT', '2011-04'], freq='M', name='idx')
    exp = TimedeltaIndex([pd.NaT] * 4, name='idx')
    tm.assert_index_equal(pd.NaT - idx, exp)
    tm.assert_index_equal(idx - pd.NaT, exp)