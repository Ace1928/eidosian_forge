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
def test_range_index_shortcut(self):
    index_name = 'foo'
    df = pd.DataFrame({'a': [1, 2, 3, 4]}, index=pd.RangeIndex(0, 8, step=2, name=index_name))
    df2 = pd.DataFrame({'a': [4, 5, 6, 7]}, index=pd.RangeIndex(0, 4))
    table = pa.Table.from_pandas(df)
    table_no_index_name = pa.Table.from_pandas(df2)
    assert len(table.schema) == 1
    result = table.to_pandas()
    tm.assert_frame_equal(result, df)
    assert isinstance(result.index, pd.RangeIndex)
    assert _pandas_api.get_rangeindex_attribute(result.index, 'step') == 2
    assert result.index.name == index_name
    result2 = table_no_index_name.to_pandas()
    tm.assert_frame_equal(result2, df2)
    assert isinstance(result2.index, pd.RangeIndex)
    assert _pandas_api.get_rangeindex_attribute(result2.index, 'step') == 1
    assert result2.index.name is None