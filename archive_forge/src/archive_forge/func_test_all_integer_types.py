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
def test_all_integer_types(self):
    data = OrderedDict()
    numpy_dtypes = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'byte', 'ubyte', 'short', 'ushort', 'intc', 'uintc', 'int_', 'uint', 'longlong', 'ulonglong']
    for dtype in numpy_dtypes:
        data[dtype] = np.arange(12, dtype=dtype)
    df = pd.DataFrame(data)
    _check_pandas_roundtrip(df)
    for np_arr in data.values():
        arr = pa.array(np_arr)
        assert arr.to_pylist() == np_arr.tolist()