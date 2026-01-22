from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
@pytest.mark.parametrize('dtype1,dtype2,expected_dtype', [('bool', 'bool', 'bool'), ('boolean', 'bool', 'boolean'), ('bool', 'boolean', 'boolean'), ('boolean', 'boolean', 'boolean')])
def test_concat_bool_types(dtype1, dtype2, expected_dtype):
    ser1 = Series([True, False], dtype=dtype1)
    ser2 = Series([False, True], dtype=dtype2)
    result = concat([ser1, ser2], ignore_index=True)
    expected = Series([True, False, False, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)