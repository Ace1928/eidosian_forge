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
def test_strided_data_import(self):
    cases = []
    columns = ['a', 'b', 'c']
    N, K = (100, 3)
    random_numbers = np.random.randn(N, K).copy() * 100
    numeric_dtypes = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8']
    for type_name in numeric_dtypes:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cases.append(random_numbers.astype(type_name))
    cases.append(np.array([random_ascii(10) for i in range(N * K)], dtype=object).reshape(N, K).copy())
    boolean_objects = np.array([True, False, True] * N, dtype=object).reshape(N, K).copy()
    boolean_objects[5] = None
    cases.append(boolean_objects)
    cases.append(np.arange('2016-01-01T00:00:00.001', N * K, dtype='datetime64[ms]').reshape(N, K).copy())
    strided_mask = (random_numbers > 0).astype(bool)[:, 0]
    for case in cases:
        df = pd.DataFrame(case, columns=columns)
        col = df['a']
        _check_pandas_roundtrip(df)
        _check_array_roundtrip(col)
        _check_array_roundtrip(col, mask=strided_mask)