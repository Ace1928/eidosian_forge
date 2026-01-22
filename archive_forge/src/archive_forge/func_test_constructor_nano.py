import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_nano(self):
    idx = period_range(start=Period(ordinal=1, freq='ns'), end=Period(ordinal=4, freq='ns'), freq='ns')
    exp = PeriodIndex([Period(ordinal=1, freq='ns'), Period(ordinal=2, freq='ns'), Period(ordinal=3, freq='ns'), Period(ordinal=4, freq='ns')], freq='ns')
    tm.assert_index_equal(idx, exp)