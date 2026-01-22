import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.parametrize('orient', ['split', 'records', 'index'])
def test_read_json_nullable_series(self, string_storage, dtype_backend, orient):
    pa = pytest.importorskip('pyarrow')
    ser = Series([1, np.nan, 3], dtype='Int64')
    out = ser.to_json(orient=orient)
    with pd.option_context('mode.string_storage', string_storage):
        result = read_json(StringIO(out), dtype_backend=dtype_backend, orient=orient, typ='series')
    expected = Series([1, np.nan, 3], dtype='Int64')
    if dtype_backend == 'pyarrow':
        from pandas.arrays import ArrowExtensionArray
        expected = Series(ArrowExtensionArray(pa.array(expected, from_pandas=True)))
    tm.assert_series_equal(result, expected)