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
@pytest.mark.parametrize('dtype', [None, np.float64, int, 'U3'])
@pytest.mark.parametrize('convert_axes', [True, False])
def test_roundtrip_str_axes(self, orient, convert_axes, dtype):
    df = DataFrame(np.zeros((200, 4)), columns=[str(i) for i in range(4)], index=[str(i) for i in range(200)], dtype=dtype)
    data = StringIO(df.to_json(orient=orient))
    result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)
    expected = df.copy()
    if not dtype:
        expected = expected.astype(np.int64)
    if convert_axes and orient in ('index', 'columns'):
        expected.columns = expected.columns.astype(np.int64)
        expected.index = expected.index.astype(np.int64)
    elif orient == 'records' and convert_axes:
        expected.columns = expected.columns.astype(np.int64)
    elif convert_axes and orient == 'split':
        expected.columns = expected.columns.astype(np.int64)
    assert_json_roundtrip_equal(result, expected, orient)