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
@pytest.mark.parametrize('val', [datetime(2000, 1, 4), datetime(2000, 1, 5)])
def test_series_comparison_scalars(self, val):
    series = Series(date_range('1/1/2000', periods=10))
    result = series > val
    expected = Series([x > val for x in series])
    tm.assert_series_equal(result, expected)