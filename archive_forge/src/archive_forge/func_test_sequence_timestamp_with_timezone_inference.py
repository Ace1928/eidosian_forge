import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
def test_sequence_timestamp_with_timezone_inference():
    pytest.importorskip('pytz')
    import pytz
    data = [datetime.datetime(2007, 7, 13, 8, 23, 34, 123456), pytz.utc.localize(datetime.datetime(2008, 1, 5, 5, 0, 0, 1000)), None, pytz.timezone('US/Eastern').localize(datetime.datetime(2006, 1, 13, 12, 34, 56, 432539)), pytz.timezone('Europe/Moscow').localize(datetime.datetime(2010, 8, 13, 5, 0, 0, 437699))]
    expected = [pa.timestamp('us', tz=None), pa.timestamp('us', tz='UTC'), pa.timestamp('us', tz=None), pa.timestamp('us', tz='US/Eastern'), pa.timestamp('us', tz='Europe/Moscow')]
    for dt, expected_type in zip(data, expected):
        prepended = [dt] + data
        arr = pa.array(prepended)
        assert arr.type == expected_type