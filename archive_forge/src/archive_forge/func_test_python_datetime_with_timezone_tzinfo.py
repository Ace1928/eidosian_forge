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
def test_python_datetime_with_timezone_tzinfo(self):
    pytz = pytest.importorskip('pytz')
    from datetime import timezone
    values = [datetime(2018, 1, 1, 12, 23, 45, tzinfo=timezone.utc)]
    df = pd.DataFrame({'datetime': values}, index=values)
    _check_pandas_roundtrip(df, preserve_index=True)
    hours = 1
    tz_timezone = timezone(timedelta(hours=hours))
    tz_pytz = pytz.FixedOffset(hours * 60)
    values = [datetime(2018, 1, 1, 12, 23, 45, tzinfo=tz_timezone)]
    values_exp = [datetime(2018, 1, 1, 12, 23, 45, tzinfo=tz_pytz)]
    df = pd.DataFrame({'datetime': values}, index=values)
    df_exp = pd.DataFrame({'datetime': values_exp}, index=values_exp)
    _check_pandas_roundtrip(df, expected=df_exp, preserve_index=True)