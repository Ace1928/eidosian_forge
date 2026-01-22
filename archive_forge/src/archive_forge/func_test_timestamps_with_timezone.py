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
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_timestamps_with_timezone(self, unit):
    if Version(pd.__version__) < Version('2.0.0') and unit != 'ns':
        pytest.skip('pandas < 2.0 only supports nanosecond datetime64')
    df = pd.DataFrame({'datetime64': np.array(['2007-07-13T01:23:34.123', '2006-01-13T12:34:56.432', '2010-08-13T05:46:57.437'], dtype=f'datetime64[{unit}]')})
    df['datetime64'] = df['datetime64'].dt.tz_localize('US/Eastern')
    _check_pandas_roundtrip(df)
    _check_series_roundtrip(df['datetime64'])
    df = pd.DataFrame({'datetime64': np.array(['2007-07-13T01:23:34.123456789', None, '2006-01-13T12:34:56.432539784', '2010-08-13T05:46:57.437699912'], dtype=f'datetime64[{unit}]')})
    df['datetime64'] = df['datetime64'].dt.tz_localize('US/Eastern')
    _check_pandas_roundtrip(df)