from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_series_str_decode():
    result = Series([b'x', b'y']).str.decode(encoding='UTF-8', errors='strict')
    expected = Series(['x', 'y'], dtype='object')
    tm.assert_series_equal(result, expected)