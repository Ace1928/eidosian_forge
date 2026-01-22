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
def test_pi_sub_pi_with_nat(self):
    rng = period_range('1/1/2000', freq='D', periods=5)
    other = rng[1:].insert(0, pd.NaT)
    assert other[1:].equals(rng[1:])
    result = rng - other
    off = rng.freq
    expected = pd.Index([pd.NaT, 0 * off, 0 * off, 0 * off, 0 * off])
    tm.assert_index_equal(result, expected)