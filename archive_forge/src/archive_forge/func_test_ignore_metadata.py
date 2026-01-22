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
def test_ignore_metadata(self):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['foo', 'bar', 'baz']}, index=['one', 'two', 'three'])
    table = pa.Table.from_pandas(df)
    result = table.to_pandas(ignore_metadata=True)
    expected = table.cast(table.schema.remove_metadata()).to_pandas()
    tm.assert_frame_equal(result, expected)