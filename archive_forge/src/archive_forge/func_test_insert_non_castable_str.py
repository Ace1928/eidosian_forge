from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_insert_non_castable_str(self, tz_aware_fixture):
    tz = tz_aware_fixture
    dti = date_range('2019-11-04', periods=3, freq='-1D', name=9, tz=tz)
    value = 'foo'
    result = dti.insert(0, value)
    expected = Index(['foo'] + list(dti), dtype=object, name=9)
    tm.assert_index_equal(result, expected)