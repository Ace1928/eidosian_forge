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
@pytest.mark.slow
@pytest.mark.large_memory
def test_from_numpy_large(self):
    target_size = 3 * 1024 ** 3
    dt = np.dtype([('x', np.float64), ('y', 'object')])
    bs = 65536 - dt.itemsize
    block = b'.' * bs
    n = target_size // (bs + dt.itemsize)
    data = np.zeros(n, dtype=dt)
    data['x'] = np.random.random_sample(n)
    data['y'] = block
    data['x'][data['x'] < 0.2] = np.nan
    ty = pa.struct([pa.field('x', pa.float64()), pa.field('y', pa.binary())])
    arr = pa.array(data, type=ty, from_pandas=True)
    arr.validate(full=True)
    assert arr.num_chunks == 2

    def iter_chunked_array(arr):
        for chunk in arr.iterchunks():
            yield from chunk

    def check(arr, data, mask=None):
        assert len(arr) == len(data)
        xs = data['x']
        ys = data['y']
        for i, obj in enumerate(iter_chunked_array(arr)):
            try:
                d = obj.as_py()
                if mask is not None and mask[i]:
                    assert d is None
                else:
                    x = xs[i]
                    if np.isnan(x):
                        assert d['x'] is None
                    else:
                        assert d['x'] == x
                    assert d['y'] == ys[i]
            except Exception:
                print('Failed at index', i)
                raise
    check(arr, data)
    del arr
    mask = np.random.random_sample(n) < 0.2
    arr = pa.array(data, type=ty, mask=mask, from_pandas=True)
    arr.validate(full=True)
    assert arr.num_chunks == 2
    check(arr, data, mask)
    del arr