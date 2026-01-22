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
def test_boolean_nulls(self):
    num_values = 100
    np.random.seed(0)
    mask = np.random.randint(0, 10, size=num_values) < 3
    values = np.random.randint(0, 10, size=num_values) < 5
    arr = pa.array(values, mask=mask)
    expected = values.astype(object)
    expected[mask] = None
    field = pa.field('bools', pa.bool_())
    schema = pa.schema([field])
    ex_frame = pd.DataFrame({'bools': expected})
    table = pa.Table.from_arrays([arr], ['bools'])
    assert table.schema.equals(schema)
    result = table.to_pandas()
    tm.assert_frame_equal(result, ex_frame)