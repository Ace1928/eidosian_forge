from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_union_same_timezone_different_units(self):
    idx1 = date_range('2000-01-01', periods=3, tz='UTC').as_unit('ms')
    idx2 = date_range('2000-01-01', periods=3, tz='UTC').as_unit('us')
    result = idx1.union(idx2)
    expected = date_range('2000-01-01', periods=3, tz='UTC').as_unit('us')
    tm.assert_index_equal(result, expected)