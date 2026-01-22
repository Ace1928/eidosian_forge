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
def test_dta_add_sub_index(self, tz_naive_fixture):
    dti = date_range('20130101', periods=3, tz=tz_naive_fixture)
    dta = dti.array
    result = dta - dti
    expected = dti - dti
    tm.assert_index_equal(result, expected)
    tdi = result
    result = dta + tdi
    expected = dti + tdi
    tm.assert_index_equal(result, expected)
    result = dta - tdi
    expected = dti - tdi
    tm.assert_index_equal(result, expected)