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
def test_float_object_nulls(self):
    arr = np.array([None, 1.5, np.float64(3.5)] * 5, dtype=object)
    df = pd.DataFrame({'floats': arr})
    expected = pd.DataFrame({'floats': pd.to_numeric(arr)})
    field = pa.field('floats', pa.float64())
    schema = pa.schema([field])
    _check_pandas_roundtrip(df, expected=expected, expected_schema=schema)