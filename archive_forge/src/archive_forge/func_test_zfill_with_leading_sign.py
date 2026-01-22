from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_zfill_with_leading_sign():
    value = Series(['-cat', '-1', '+dog'])
    expected = Series(['-0cat', '-0001', '+0dog'])
    tm.assert_series_equal(value.str.zfill(5), expected)