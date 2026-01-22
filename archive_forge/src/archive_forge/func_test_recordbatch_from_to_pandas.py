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
def test_recordbatch_from_to_pandas():
    data = pd.DataFrame({'c1': np.array([1, 2, 3, 4, 5], dtype='int64'), 'c2': np.array([1, 2, 3, 4, 5], dtype='uint32'), 'c3': np.random.randn(5), 'c4': ['foo', 'bar', None, 'baz', 'qux'], 'c5': [False, True, False, True, False]})
    batch = pa.RecordBatch.from_pandas(data)
    result = batch.to_pandas()
    tm.assert_frame_equal(data, result)