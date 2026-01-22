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
def test_fixed_size_bytes(self):
    values = [b'foo', None, bytearray(b'bar'), None, None, b'hey']
    df = pd.DataFrame({'strings': values})
    schema = pa.schema([pa.field('strings', pa.binary(3))])
    table = pa.Table.from_pandas(df, schema=schema)
    assert table.schema[0].type == schema[0].type
    assert table.schema[0].name == schema[0].name
    result = table.to_pandas()
    tm.assert_frame_equal(result, df)