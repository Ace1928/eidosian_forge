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
def test_integer_no_nulls(self):
    data = OrderedDict()
    fields = []
    numpy_dtypes = [('i1', pa.int8()), ('i2', pa.int16()), ('i4', pa.int32()), ('i8', pa.int64()), ('u1', pa.uint8()), ('u2', pa.uint16()), ('u4', pa.uint32()), ('u8', pa.uint64()), ('longlong', pa.int64()), ('ulonglong', pa.uint64())]
    num_values = 100
    for dtype, arrow_dtype in numpy_dtypes:
        info = np.iinfo(dtype)
        values = np.random.randint(max(info.min, np.iinfo(np.int_).min), min(info.max, np.iinfo(np.int_).max), size=num_values)
        data[dtype] = values.astype(dtype)
        fields.append(pa.field(dtype, arrow_dtype))
    df = pd.DataFrame(data)
    schema = pa.schema(fields)
    _check_pandas_roundtrip(df, expected_schema=schema)