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
def test_index_metadata_field_name(self):
    df = pd.DataFrame([(1, 'a', 3.1), (2, 'b', 2.2), (3, 'c', 1.3)], index=pd.MultiIndex.from_arrays([['c', 'b', 'a'], [3, 2, 1]], names=[None, 'foo']), columns=['a', None, '__index_level_0__'])
    with pytest.warns(UserWarning):
        t = pa.Table.from_pandas(df, preserve_index=True)
    js = t.schema.pandas_metadata
    col1, col2, col3, idx0, foo = js['columns']
    assert col1['name'] == 'a'
    assert col1['name'] == col1['field_name']
    assert col2['name'] is None
    assert col2['field_name'] == 'None'
    assert col3['name'] == '__index_level_0__'
    assert col3['name'] == col3['field_name']
    idx0_descr, foo_descr = js['index_columns']
    assert idx0_descr == '__index_level_0__'
    assert idx0['field_name'] == idx0_descr
    assert idx0['name'] is None
    assert foo_descr == 'foo'
    assert foo['field_name'] == foo_descr
    assert foo['name'] == foo_descr