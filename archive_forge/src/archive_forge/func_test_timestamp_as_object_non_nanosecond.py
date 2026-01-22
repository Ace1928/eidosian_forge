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
@pytest.mark.parametrize('resolution', ['s', 'ms', 'us'])
@pytest.mark.parametrize('tz', [None, 'America/New_York'])
@pytest.mark.parametrize('dt', [datetime(1553, 1, 1), datetime(2020, 1, 1)])
def test_timestamp_as_object_non_nanosecond(resolution, tz, dt):
    arr = pa.array([dt], type=pa.timestamp(resolution, tz=tz))
    table = pa.table({'a': arr})
    for result in [arr.to_pandas(timestamp_as_object=True), table.to_pandas(timestamp_as_object=True)['a']]:
        assert result.dtype == object
        assert isinstance(result[0], datetime)
        if tz:
            assert result[0].tzinfo is not None
            expected = result[0].tzinfo.fromutc(dt)
        else:
            assert result[0].tzinfo is None
            expected = dt
        assert result[0] == expected