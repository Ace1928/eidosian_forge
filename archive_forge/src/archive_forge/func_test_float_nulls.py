import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def test_float_nulls(self):
    num_values = 100
    null_mask = np.random.randint(0, 10, size=num_values) < 3
    dtypes = [('f2', pa.float16()), ('f4', pa.float32()), ('f8', pa.float64())]
    names = ['f2', 'f4', 'f8']
    expected_cols = []
    arrays = []
    fields = []
    for name, arrow_dtype in dtypes:
        values = np.random.randn(num_values).astype(name)
        arr = pa.array(values, from_pandas=True, mask=null_mask)
        arrays.append(arr)
        fields.append(pa.field(name, arrow_dtype))
        values[null_mask] = np.nan
        expected_cols.append(values)
    ex_frame = pd.DataFrame(dict(zip(names, expected_cols)), columns=names)
    table = pa.Table.from_arrays(arrays, names)
    assert table.schema.equals(pa.schema(fields))
    result = table.to_pandas()
    tm.assert_frame_equal(result, ex_frame)