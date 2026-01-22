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
@pytest.mark.parametrize('other', ['foo', -1, 99, 4.0, object(), timedelta(days=2), datetime(2001, 1, 1).date(), None, np.nan])
def test_dt64arr_cmp_scalar_invalid(self, other, tz_naive_fixture, box_with_array):
    tz = tz_naive_fixture
    rng = date_range('1/1/2000', periods=10, tz=tz)
    dtarr = tm.box_expected(rng, box_with_array)
    assert_invalid_comparison(dtarr, other, box_with_array)