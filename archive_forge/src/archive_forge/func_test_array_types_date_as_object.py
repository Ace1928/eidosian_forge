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
@pytest.mark.parametrize('coerce_to_ns,expected_dtype', [(False, 'datetime64[ms]'), (True, 'datetime64[ns]')])
def test_array_types_date_as_object(self, coerce_to_ns, expected_dtype):
    data = [date(2000, 1, 1), None, date(1970, 1, 1), date(2040, 2, 26)]
    expected_days = np.array(['2000-01-01', None, '1970-01-01', '2040-02-26'], dtype='datetime64[D]')
    if Version(pd.__version__) < Version('2.0.0'):
        expected_dtype = 'datetime64[ns]'
    expected = np.array(['2000-01-01', None, '1970-01-01', '2040-02-26'], dtype=expected_dtype)
    objects = [pa.array(data), pa.chunked_array([data])]
    for obj in objects:
        result = obj.to_pandas(coerce_temporal_nanoseconds=coerce_to_ns)
        expected_obj = expected_days.astype(object)
        assert result.dtype == expected_obj.dtype
        npt.assert_array_equal(result, expected_obj)
        result = obj.to_pandas(date_as_object=False, coerce_temporal_nanoseconds=coerce_to_ns)
        assert result.dtype == expected.dtype
        npt.assert_array_equal(result, expected)