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
def test_date_infer(self):
    df = pd.DataFrame({'date': [date(2000, 1, 1), None, date(1970, 1, 1), date(2040, 2, 26)]})
    table = pa.Table.from_pandas(df, preserve_index=False)
    field = pa.field('date', pa.date32())
    expected_schema = pa.schema([field], metadata=table.schema.metadata)
    assert table.schema.equals(expected_schema)
    result = table.to_pandas()
    tm.assert_frame_equal(result, df)