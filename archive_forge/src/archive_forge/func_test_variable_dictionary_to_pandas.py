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
def test_variable_dictionary_to_pandas():
    np.random.seed(12345)
    d1 = pa.array(random_strings(100, 32), type='string')
    d2 = pa.array(random_strings(100, 16), type='string')
    d3 = pa.array(random_strings(10000, 10), type='string')
    a1 = pa.DictionaryArray.from_arrays(np.random.randint(0, len(d1), size=1000, dtype='i4'), d1)
    a2 = pa.DictionaryArray.from_arrays(np.random.randint(0, len(d2), size=1000, dtype='i4'), d2)
    a3 = pa.DictionaryArray.from_arrays(np.random.randint(0, len(d3), size=1000, dtype='i4'), d3)
    i4 = pa.array(np.random.randint(0, len(d3), size=1000, dtype='i4'), mask=np.random.rand(1000) < 0.1)
    a4 = pa.DictionaryArray.from_arrays(i4, d3)
    expected_dict = pa.concat_arrays([d1, d2, d3])
    a = pa.chunked_array([a1, a2, a3, a4])
    a_dense = pa.chunked_array([a1.cast('string'), a2.cast('string'), a3.cast('string'), a4.cast('string')])
    result = a.to_pandas()
    result_dense = a_dense.to_pandas()
    assert (result.cat.categories == expected_dict.to_pandas()).all()
    expected_dense = result.astype('str')
    expected_dense[result_dense.isnull()] = None
    tm.assert_series_equal(result_dense, expected_dense)