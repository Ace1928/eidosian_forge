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
def test_add_na_handling(self):
    ser = Series([Decimal('1.3'), Decimal('2.3')], index=[date(2012, 1, 1), date(2012, 1, 2)])
    result = ser + ser.shift(1)
    result2 = ser.shift(1) + ser
    assert isna(result.iloc[0])
    assert isna(result2.iloc[0])