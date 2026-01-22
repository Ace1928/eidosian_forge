from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_constructor_period_values_mismatched_dtype(self):
    pi = period_range('2016-01-01', periods=3, freq='D')
    result = Index(pi, dtype='category')
    expected = CategoricalIndex(pi)
    tm.assert_index_equal(result, expected)