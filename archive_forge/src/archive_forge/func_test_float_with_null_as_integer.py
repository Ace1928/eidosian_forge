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
def test_float_with_null_as_integer(self):
    s = pd.Series([np.nan, 1.0, 2.0, np.nan])
    types = [pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]
    for ty in types:
        result = pa.array(s, type=ty)
        expected = pa.array([None, 1, 2, None], type=ty)
        assert result.equals(expected)
        df = pd.DataFrame({'has_nulls': s})
        schema = pa.schema([pa.field('has_nulls', ty)])
        result = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        assert result[0].chunk(0).equals(expected)