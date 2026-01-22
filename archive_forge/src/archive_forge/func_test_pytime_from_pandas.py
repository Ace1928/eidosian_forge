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
def test_pytime_from_pandas(self):
    pytimes = [time(1, 2, 3, 1356), time(4, 5, 6, 1356)]
    t1 = pa.time64('us')
    aobjs = np.array(pytimes + [None], dtype=object)
    parr = pa.array(aobjs)
    assert parr.type == t1
    assert parr[0].as_py() == pytimes[0]
    assert parr[1].as_py() == pytimes[1]
    assert parr[2].as_py() is None
    df = pd.DataFrame({'times': aobjs})
    batch = pa.RecordBatch.from_pandas(df)
    assert batch[0].equals(parr)
    arr = np.array([_pytime_to_micros(v) for v in pytimes], dtype='int64')
    a1 = pa.array(arr, type=pa.time64('us'))
    assert a1[0].as_py() == pytimes[0]
    a2 = pa.array(arr * 1000, type=pa.time64('ns'))
    assert a2[0].as_py() == pytimes[0]
    a3 = pa.array((arr / 1000).astype('i4'), type=pa.time32('ms'))
    assert a3[0].as_py() == pytimes[0].replace(microsecond=1000)
    a4 = pa.array((arr / 1000000).astype('i4'), type=pa.time32('s'))
    assert a4[0].as_py() == pytimes[0].replace(microsecond=0)