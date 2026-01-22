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
def test_date_objects_typed(self):
    arr = np.array([date(2017, 4, 3), None, date(2017, 4, 4), date(2017, 4, 5)], dtype=object)
    arr_i4 = np.array([17259, -1, 17260, 17261], dtype='int32')
    arr_i8 = arr_i4.astype('int64') * 86400000
    mask = np.array([False, True, False, False])
    t32 = pa.date32()
    t64 = pa.date64()
    a32 = pa.array(arr, type=t32)
    a64 = pa.array(arr, type=t64)
    a32_expected = pa.array(arr_i4, mask=mask, type=t32)
    a64_expected = pa.array(arr_i8, mask=mask, type=t64)
    assert a32.equals(a32_expected)
    assert a64.equals(a64_expected)
    colnames = ['date32', 'date64']
    table = pa.Table.from_arrays([a32, a64], colnames)
    ex_values = np.array(['2017-04-03', '2017-04-04', '2017-04-04', '2017-04-05'], dtype='datetime64[D]')
    ex_values[1] = pd.NaT.value
    ex_datetime64ms = ex_values.astype('datetime64[ms]')
    expected_pandas = pd.DataFrame({'date32': ex_datetime64ms, 'date64': ex_datetime64ms}, columns=colnames)
    table_pandas = table.to_pandas(date_as_object=False)
    tm.assert_frame_equal(table_pandas, expected_pandas)
    table_pandas_objects = table.to_pandas()
    ex_objects = ex_values.astype('object')
    expected_pandas_objects = pd.DataFrame({'date32': ex_objects, 'date64': ex_objects}, columns=colnames)
    tm.assert_frame_equal(table_pandas_objects, expected_pandas_objects)