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
@pytest.mark.large_memory
def test_bytes_exceed_2gb(self):
    v1 = b'x' * 100000000
    v2 = b'x' * 147483646
    df = pd.DataFrame({'strings': [v1] * 20 + [v2] + ['x'] * 20})
    arr = pa.array(df['strings'])
    assert isinstance(arr, pa.ChunkedArray)
    assert arr.num_chunks == 2
    arr = None
    table = pa.Table.from_pandas(df)
    assert table[0].num_chunks == 2