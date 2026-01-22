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
def test_empty_series_add_sub(self, box_with_array):
    a = Series(dtype='M8[ns]')
    b = Series(dtype='m8[ns]')
    a = box_with_array(a)
    b = box_with_array(b)
    tm.assert_equal(a, a + b)
    tm.assert_equal(a, a - b)
    tm.assert_equal(a, b + a)
    msg = 'cannot subtract'
    with pytest.raises(TypeError, match=msg):
        b - a