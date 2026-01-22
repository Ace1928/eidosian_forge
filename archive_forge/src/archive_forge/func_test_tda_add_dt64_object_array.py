from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_tda_add_dt64_object_array(self, box_with_array, tz_naive_fixture):
    box = box_with_array
    dti = pd.date_range('2016-01-01', periods=3, tz=tz_naive_fixture)
    dti = dti._with_freq(None)
    tdi = dti - dti
    obj = tm.box_expected(tdi, box)
    other = tm.box_expected(dti, box)
    with tm.assert_produces_warning(PerformanceWarning):
        result = obj + other.astype(object)
    tm.assert_equal(result, other.astype(object))