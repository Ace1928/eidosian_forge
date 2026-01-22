from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_mul_empty_int_corner_case(self):
    s1 = Series([], [], dtype=np.int32)
    s2 = Series({'x': 0.0})
    tm.assert_series_equal(s1 * s2, Series([np.nan], index=['x']))