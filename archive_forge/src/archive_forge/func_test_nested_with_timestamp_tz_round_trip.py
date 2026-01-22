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
def test_nested_with_timestamp_tz_round_trip():
    ts = pd.Timestamp.now()
    ts_dt = ts.to_pydatetime()
    arr = pa.array([ts_dt], type=pa.timestamp('us', tz='America/New_York'))
    struct = pa.StructArray.from_arrays([arr, arr], ['start', 'stop'])
    result = struct.to_pandas()
    restored = pa.array(result)
    assert restored.equals(struct)