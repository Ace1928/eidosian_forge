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
def test_table_column_subset_metadata(self):
    for index in [pd.Index(['a', 'b', 'c'], name='index'), pd.date_range('2017-01-01', periods=3, tz='Europe/Brussels')]:
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]}, index=index)
        table = pa.Table.from_pandas(df)
        table_subset = table.remove_column(1)
        result = table_subset.to_pandas()
        expected = df[['a']]
        if isinstance(df.index, pd.DatetimeIndex):
            df.index.freq = None
        tm.assert_frame_equal(result, expected)
        table_subset2 = table_subset.remove_column(1)
        result = table_subset2.to_pandas()
        tm.assert_frame_equal(result, df[['a']].reset_index(drop=True))