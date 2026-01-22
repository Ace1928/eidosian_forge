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
def test_timestamp_to_pandas_coerces_to_ns(self):
    if Version(pd.__version__) >= Version('2.0.0'):
        pytest.skip('pandas >= 2.0 supports non-nanosecond datetime64')
    arr = pa.array([1, 2, 3], pa.timestamp('ms'))
    expected = pd.Series(pd.to_datetime([1, 2, 3], unit='ms'))
    s = arr.to_pandas()
    tm.assert_series_equal(s, expected)
    arr = pa.chunked_array([arr])
    s = arr.to_pandas()
    tm.assert_series_equal(s, expected)