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
def test_mismatch_metadata_schema(self):
    df = pd.DataFrame({'datetime': pd.date_range('2020-01-01', periods=3)})
    table = pa.Table.from_pandas(df)
    new_col = table['datetime'].cast(pa.timestamp('ns', tz='UTC'))
    new_table1 = table.set_column(0, pa.field('datetime', new_col.type), new_col)
    schema = pa.schema([('datetime', pa.timestamp('ns', tz='UTC'))])
    new_table2 = pa.Table.from_pandas(df, schema=schema)
    expected = df.copy()
    expected['datetime'] = expected['datetime'].dt.tz_localize('UTC')
    for new_table in [new_table1, new_table2]:
        assert new_table.schema.pandas_metadata is not None
        result = new_table.to_pandas()
        tm.assert_frame_equal(result, expected)