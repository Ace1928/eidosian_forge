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
def test_categorical_column_index(self):
    df = pd.DataFrame([(1, 'a', 2.0), (2, 'b', 3.0), (3, 'c', 4.0)], columns=pd.Index(list('def'), dtype='category'))
    t = pa.Table.from_pandas(df, preserve_index=True)
    js = t.schema.pandas_metadata
    column_indexes, = js['column_indexes']
    assert column_indexes['name'] is None
    assert column_indexes['pandas_type'] == 'categorical'
    assert column_indexes['numpy_type'] == 'int8'
    md = column_indexes['metadata']
    assert md['num_categories'] == 3
    assert md['ordered'] is False