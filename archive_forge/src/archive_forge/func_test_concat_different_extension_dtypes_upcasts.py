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
def test_concat_different_extension_dtypes_upcasts(self):
    a = Series(pd.array([1, 2], dtype='Int64'))
    b = Series(to_decimal([1, 2]))
    result = concat([a, b], ignore_index=True)
    expected = Series([1, 2, Decimal(1), Decimal(2)], dtype=object)
    tm.assert_series_equal(result, expected)