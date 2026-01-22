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
def test_timestamps_notimezone_no_nulls(self):
    df = pd.DataFrame({'datetime64': np.array(['2007-07-13T01:23:34.123456789', '2006-01-13T12:34:56.432539784', '2010-08-13T05:46:57.437699912'], dtype='datetime64[ns]')})
    field = pa.field('datetime64', pa.timestamp('ns'))
    schema = pa.schema([field])
    _check_pandas_roundtrip(df, expected_schema=schema)