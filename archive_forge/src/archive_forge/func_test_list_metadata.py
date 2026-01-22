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
def test_list_metadata(self):
    df = pd.DataFrame({'data': [[1], [2, 3, 4], [5] * 7]})
    schema = pa.schema([pa.field('data', type=pa.list_(pa.int64()))])
    table = pa.Table.from_pandas(df, schema=schema)
    js = table.schema.pandas_metadata
    assert 'mixed' not in js
    data_column = js['columns'][0]
    assert data_column['pandas_type'] == 'list[int64]'
    assert data_column['numpy_type'] == 'object'