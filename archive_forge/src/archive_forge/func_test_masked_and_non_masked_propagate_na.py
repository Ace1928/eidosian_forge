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
def test_masked_and_non_masked_propagate_na(self):
    ser1 = Series([0, np.nan], dtype='float')
    ser2 = Series([0, 1], dtype='Int64')
    result = ser1 * ser2
    expected = Series([0, pd.NA], dtype='Float64')
    tm.assert_series_equal(result, expected)