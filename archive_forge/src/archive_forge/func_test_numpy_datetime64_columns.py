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
def test_numpy_datetime64_columns(self):
    datetime64_ns = np.array(['2007-07-13T01:23:34.123456789', None, '2006-01-13T12:34:56.432539784', '2010-08-13T05:46:57.437699912'], dtype='datetime64[ns]')
    _check_array_from_pandas_roundtrip(datetime64_ns)
    datetime64_us = np.array(['2007-07-13T01:23:34.123456', None, '2006-01-13T12:34:56.432539', '2010-08-13T05:46:57.437699'], dtype='datetime64[us]')
    _check_array_from_pandas_roundtrip(datetime64_us)
    datetime64_ms = np.array(['2007-07-13T01:23:34.123', None, '2006-01-13T12:34:56.432', '2010-08-13T05:46:57.437'], dtype='datetime64[ms]')
    _check_array_from_pandas_roundtrip(datetime64_ms)
    datetime64_s = np.array(['2007-07-13T01:23:34', None, '2006-01-13T12:34:56', '2010-08-13T05:46:57'], dtype='datetime64[s]')
    _check_array_from_pandas_roundtrip(datetime64_s)