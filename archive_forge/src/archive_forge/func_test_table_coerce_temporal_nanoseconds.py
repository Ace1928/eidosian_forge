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
@pytest.mark.parametrize('arrow_type', [pa.date32(), pa.date64(), pa.timestamp('s'), pa.timestamp('ms'), pa.timestamp('us'), pa.timestamp('ns'), pa.timestamp('s', 'UTC'), pa.timestamp('ms', 'UTC'), pa.timestamp('us', 'UTC'), pa.timestamp('ns', 'UTC')])
def test_table_coerce_temporal_nanoseconds(self, arrow_type):
    data = [date(2000, 1, 1), datetime(2001, 1, 1)]
    schema = pa.schema([pa.field('date', arrow_type)])
    expected_df = pd.DataFrame({'date': data})
    table = pa.table([pa.array(data)], schema=schema)
    result_df = table.to_pandas(coerce_temporal_nanoseconds=True, date_as_object=False)
    expected_tz = None
    if hasattr(arrow_type, 'tz') and arrow_type.tz is not None:
        expected_tz = 'UTC'
    expected_type = pa.timestamp('ns', expected_tz).to_pandas_dtype()
    tm.assert_frame_equal(result_df, expected_df.astype(expected_type))