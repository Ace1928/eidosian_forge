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
def test_table_from_pandas_schema_index_columns():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
    schema = pa.schema([('a', pa.int64()), ('b', pa.float64()), ('index', pa.int64())])
    with pytest.raises(KeyError, match="name 'index' present in the"):
        pa.Table.from_pandas(df, schema=schema)
    df.index.name = 'index'
    _check_pandas_roundtrip(df, schema=schema, preserve_index=True, expected_schema=schema)
    with pytest.raises(ValueError, match="'preserve_index=False' was"):
        pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    with pytest.raises(ValueError, match="name 'index' is present in the schema, but it is a RangeIndex"):
        pa.Table.from_pandas(df, schema=schema, preserve_index=None)
    df.index = pd.Index([0, 1, 2], name='index')
    _check_pandas_roundtrip(df, schema=schema, preserve_index=None, expected_schema=schema)
    _check_pandas_roundtrip(df, schema=schema, preserve_index=True, expected_schema=schema)
    schema = pa.schema([('index', pa.int64()), ('a', pa.int64()), ('b', pa.float64())])
    _check_pandas_roundtrip(df, schema=schema, preserve_index=None, expected_schema=schema)
    _check_pandas_roundtrip(df, schema=schema, preserve_index=True, expected_schema=schema)
    schema = pa.schema([('a', pa.int64()), ('b', pa.float64())])
    expected = df.copy()
    expected = expected.reset_index(drop=True)
    _check_pandas_roundtrip(df, schema=schema, preserve_index=None, expected_schema=schema, expected=expected)
    _check_pandas_roundtrip(df, schema=schema, preserve_index=True, expected_schema=schema, expected=expected)
    df.index = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1)], names=['level1', 'level2'])
    schema = pa.schema([('level1', pa.string()), ('level2', pa.int64()), ('a', pa.int64()), ('b', pa.float64())])
    _check_pandas_roundtrip(df, schema=schema, preserve_index=True, expected_schema=schema)
    _check_pandas_roundtrip(df, schema=schema, preserve_index=None, expected_schema=schema)
    schema = pa.schema([('level2', pa.int64()), ('a', pa.int64()), ('b', pa.float64())])
    expected = df.copy()
    expected = expected.reset_index('level1', drop=True)
    _check_pandas_roundtrip(df, schema=schema, preserve_index=True, expected_schema=schema, expected=expected)
    _check_pandas_roundtrip(df, schema=schema, preserve_index=None, expected_schema=schema, expected=expected)