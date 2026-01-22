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
@pytest.mark.parametrize('timezone', [None, 'UTC', 'Etc/GMT-1', 'Europe/Budapest'])
def test_pyarrow_ignore_timezone_environment_variable(monkeypatch, timezone):
    pytest.importorskip('pytz')
    import pytz
    monkeypatch.setenv('PYARROW_IGNORE_TIMEZONE', '1')
    data = [datetime.datetime(2007, 7, 13, 8, 23, 34, 123456), pytz.utc.localize(datetime.datetime(2008, 1, 5, 5, 0, 0, 1000)), pytz.timezone('US/Eastern').localize(datetime.datetime(2006, 1, 13, 12, 34, 56, 432539)), pytz.timezone('Europe/Moscow').localize(datetime.datetime(2010, 8, 13, 5, 0, 0, 437699))]
    expected = [dt.replace(tzinfo=None) for dt in data]
    if timezone is not None:
        tzinfo = pytz.timezone(timezone)
        expected = [tzinfo.fromutc(dt) for dt in expected]
    ty = pa.timestamp('us', tz=timezone)
    arr = pa.array(data, type=ty)
    assert arr.to_pylist() == expected