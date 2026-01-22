from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('func', ['maximum', 'minimum'])
def test_numpy_reduction_with_tz_aware_dtype(self, tz_aware_fixture, func):
    tz = tz_aware_fixture
    arg = pd.to_datetime(['2019']).tz_localize(tz)
    expected = Series(arg)
    result = getattr(np, func)(expected, expected)
    tm.assert_series_equal(result, expected)