import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_coerce_pyarrow_backend():
    pa = pytest.importorskip('pyarrow')
    ser = Series(list('12x'), dtype=ArrowDtype(pa.string()))
    result = to_numeric(ser, errors='coerce', dtype_backend='pyarrow')
    expected = Series([1, 2, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)