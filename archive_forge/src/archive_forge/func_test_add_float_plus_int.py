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
def test_add_float_plus_int(self, datetime_series):
    int_ts = datetime_series.astype(int)[:-5]
    added = datetime_series + int_ts
    expected = Series(datetime_series.values[:-5] + int_ts.values, index=datetime_series.index[:-5], name='ts')
    tm.assert_series_equal(added[:-5], expected)