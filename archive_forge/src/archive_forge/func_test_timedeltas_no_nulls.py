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
def test_timedeltas_no_nulls(self, unit):
    if Version(pd.__version__) < Version('2.0.0'):
        unit = 'ns'
    df = pd.DataFrame({'timedelta64': np.array([0, 3600000000000, 7200000000000], dtype=f'timedelta64[{unit}]')})
    field = pa.field('timedelta64', pa.duration(unit))
    schema = pa.schema([field])
    _check_pandas_roundtrip(df, expected_schema=schema)