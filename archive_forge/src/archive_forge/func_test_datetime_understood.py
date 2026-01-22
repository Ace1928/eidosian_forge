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
def test_datetime_understood(self, unit):
    series = Series(date_range('2012-01-01', periods=3, unit=unit))
    offset = pd.offsets.DateOffset(days=6)
    result = series - offset
    exp_dti = pd.to_datetime(['2011-12-26', '2011-12-27', '2011-12-28']).as_unit(unit)
    expected = Series(exp_dti)
    tm.assert_series_equal(result, expected)