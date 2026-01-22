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
@h.given(st.none() | past.timezones)
@h.settings(deadline=None)
def test_python_datetime_with_pytz_timezone(self, tz):
    if str(tz) in ['build/etc/localtime', 'Factory']:
        pytest.skip('Localtime timezone not supported')
    values = [datetime(2018, 1, 1, 12, 23, 45, tzinfo=tz)]
    df = pd.DataFrame({'datetime': values})
    _check_pandas_roundtrip(df, check_dtype=False)