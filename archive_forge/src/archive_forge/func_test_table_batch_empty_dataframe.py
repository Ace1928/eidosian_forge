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
def test_table_batch_empty_dataframe(self):
    df = pd.DataFrame({})
    _check_pandas_roundtrip(df, preserve_index=None)
    _check_pandas_roundtrip(df, preserve_index=None, as_batch=True)
    expected = pd.DataFrame(columns=pd.Index([]))
    _check_pandas_roundtrip(df, expected, preserve_index=False)
    _check_pandas_roundtrip(df, expected, preserve_index=False, as_batch=True)
    df2 = pd.DataFrame({}, index=[0, 1, 2])
    _check_pandas_roundtrip(df2, preserve_index=True)
    _check_pandas_roundtrip(df2, as_batch=True, preserve_index=True)